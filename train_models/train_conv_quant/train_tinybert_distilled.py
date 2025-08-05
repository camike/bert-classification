################################################
# 本文件用于蒸馏微调TinyBert模型（未设置验证集）
#
# 功能说明：
# - 输入: 训练集；教师模型路径
# - 输出: 蒸馏微调后的TinyBert模型
#
# 需要修改的配置项（共4处需要修改，已在代码中用注释"x"标记）：
# 1. 教师模型、训练集、模型保存路径（3处需要修改）：第54-56行
# 2. 标签映射（1处需要修改）: 第59行
################################################




import os
import torch
import numpy as np
import random
import csv
from tqdm.auto import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    PreTrainedModel
)
from torch.nn import KLDivLoss
from torch.utils.data import Dataset

# ====================== 固定随机种子 ======================
SEED = 42
def set_seed(seed: int = SEED):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# ====================== 配置 ======================
class Config:
    """集中管理所有配置参数"""
    TEACHER_MODEL_PATH = "trained_model_bert" # 修改为教师模型路径（1）
    DATASET_PATH = "dataset.csv"        # 修改为训练数据路径（2）
    OUTPUT_DIR = "trained_model_tinybert_distilled"      # 修改为模型输出目录（3）
    MAX_LENGTH = 128                    # 最大序列长度
    
    # 修改标签映射（4）
    LABEL_MAPPING = {
        0: "其他",
        1: "爱奇艺",
        2: "飞书", 
        3: "鲁大师"
    }
    
    # 学生模型配置 (TinyBERT规格)
    STUDENT_CONFIG = {
        "hidden_size": 312,
        "num_hidden_layers": 4,
        "num_attention_heads": 12,
        "intermediate_size": 1200,
        "vocab_size": 30522,
        "num_labels": len(LABEL_MAPPING),
        "id2label": LABEL_MAPPING,
        "label2id": {v:k for k,v in LABEL_MAPPING.items()}
    }
    
    # 训练参数 (强制使用FP32)
    TRAINING_ARGS = {
        "per_device_train_batch_size": 32,
        "num_train_epochs": 10, 
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "logging_steps": 100,
        "save_steps": 500,
        "output_dir": OUTPUT_DIR,
        "save_total_limit": 2,
        "disable_tqdm": False,
        "fp16": False,                  # 禁用混合精度
        "bf16": False,                  # 禁用bfloat16
        "tf32": False                   # 禁用TF32
    }

# ====================== 数据集类 ======================
class DistillationDataset(Dataset):
    """用于知识蒸馏的数据集类"""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer, 
                 file_path: str, 
                 max_length: int = Config.MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Tuple[str, int]] = []
        self._load_data(file_path)
    
    def _load_data(self, file_path: str):
        """加载和预处理数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件 {file_path} 不存在")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV文件为空")
            
            total_rows = sum(1 for _ in reader)
            f.seek(0)
            next(reader)
            
            valid_count = 0
            with tqdm(total=total_rows, desc="📂 加载数据集", unit="样本") as pbar:
                for i, row in enumerate(reader, 1):
                    try:
                        if len(row) < 2:
                            raise ValueError("行缺少列")
                            
                        label = int(row[0].strip())
                        text = row[1].strip()
                        
                        if not text:
                            raise ValueError("文本为空")
                        if label not in Config.LABEL_MAPPING:
                            raise ValueError(f"无效标签 {label}")
                            
                        self.examples.append((text, label))
                        valid_count += 1
                    except Exception as e:
                        tqdm.write(f"⚠️ 跳过第 {i} 行: {e} - 内容: {row}")
                    finally:
                        pbar.update(1)
        
        print(f"\n✅ 成功加载 {valid_count}/{total_rows} 个样本 (跳过 {total_rows - valid_count} 个无效行)")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ====================== 蒸馏损失函数 ======================
class DistillationLoss:
    """知识蒸馏损失函数，结合KL散度和交叉熵"""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def __call__(self, 
                 student_logits: torch.Tensor, 
                 teacher_logits: torch.Tensor, 
                 labels: torch.Tensor) -> torch.Tensor:
        soft_teacher = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        kld_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        return self.alpha * kld_loss + (1. - self.alpha) * ce_loss

# ====================== 蒸馏Trainer ======================
class DistillationTrainer(Trainer):
    """自定义蒸馏Trainer"""
    
    def __init__(self, 
                 teacher_model: PreTrainedModel,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.distillation_loss = DistillationLoss()
        self.progress_bar = None
        self.start_time = datetime.now()
    
    def compute_loss(self, 
                    model: torch.nn.Module, 
                    inputs: Dict[str, torch.Tensor], 
                    return_outputs: bool = False) -> torch.Tensor:
        """计算蒸馏损失"""
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            teacher_logits = teacher_outputs.logits
        
        loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            inputs['labels']
        )
        
        return (loss, outputs) if return_outputs else loss
    
    def _inner_training_loop(self, *args, **kwargs):
        """自定义训练循环"""
        total_steps = self.args.max_steps if self.args.max_steps else \
            self.args.num_train_epochs * len(self.train_dataset) // self.args.per_device_train_batch_size
        
        self.progress_bar = tqdm(total=total_steps, desc="🚀 训练进度", unit="step")
        self.progress_bar.set_postfix({
            "epoch": "0",
            "loss": "0.0000",
            "lr": "0.0000",
            "elapsed": "0:00:00"
        })
        
        try:
            result = super()._inner_training_loop(*args, **kwargs)
        except Exception as e:
            self.progress_bar.close()
            raise e
        finally:
            if self.progress_bar is not None:
                self.progress_bar.close()
        
        return result
    
    def training_step(self, 
                     model: torch.nn.Module, 
                     inputs: Dict[str, torch.Tensor], 
                     num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """单步训练"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        loss.backward()

        if hasattr(self, 'progress_bar'):
            elapsed = datetime.now() - self.start_time
            current_lr = self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 0
            self.progress_bar.set_postfix({
                "epoch": f"{self.state.epoch:.1f}",
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.4e}",
                "elapsed": str(elapsed).split('.')[0]
            })
            self.progress_bar.update(1)

        return loss.detach()
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型"""
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        with tqdm(total=4, desc="💾 保存模型", leave=False) as pbar:
            self.model.save_pretrained(output_dir)
            pbar.update(1)
            
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            pbar.update(1)
            
            torch.save(self.state, os.path.join(output_dir, "trainer_state.pt"))
            pbar.update(1)
            
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                f.write(str(self.args.to_dict()))
            pbar.update(1)

# ====================== 主函数 ======================
def main():
    """主执行函数"""
    print("="*50)
    print(f"🧪 知识蒸馏实验 - {len(Config.LABEL_MAPPING)}分类任务")
    print(f"🏷️ 标签映射: {Config.LABEL_MAPPING}")
    print(f"🎲 随机种子: {SEED}")
    print(f"🕒 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # 设备检测
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device.upper()}")
    
    # 1. 初始化组件
    print("🔧 初始化组件...")
    try:
        tokenizer = BertTokenizer.from_pretrained(Config.TEACHER_MODEL_PATH)
        
        teacher_model = BertForSequenceClassification.from_pretrained(
            Config.TEACHER_MODEL_PATH,
            num_labels=len(Config.LABEL_MAPPING),
            id2label=Config.LABEL_MAPPING,
            label2id={v:k for k,v in Config.LABEL_MAPPING.items()}
        ).to(device)
        
        student_model = BertForSequenceClassification(
            BertConfig(**Config.STUDENT_CONFIG)
        ).to(device)
        
        print("✅ 组件初始化完成")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise
    
    # 2. 加载数据集
    print("\n📂 加载数据集...")
    try:
        full_dataset = DistillationDataset(tokenizer, Config.DATASET_PATH)
        if len(full_dataset) == 0:
            raise ValueError("数据集为空，请检查数据文件")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        raise
    
    # 3. 训练配置
    training_args = TrainingArguments(
        **Config.TRAINING_ARGS,
        eval_strategy="no",
        load_best_model_at_end=False,
        logging_dir=os.path.join(Config.OUTPUT_DIR, "logs"),
        report_to="none",
        no_cuda=(device != "cuda")  # 非CUDA设备禁用CUDA
    )
    
    # 4. 开始蒸馏
    print("\n" + "="*50)
    print(f"🔥 开始蒸馏 ({len(full_dataset)}个样本)")
    print(f"👨‍🏫 教师模型: {Config.TEACHER_MODEL_PATH}")
    print(f"👨‍🎓 学生模型架构: hidden_size={Config.STUDENT_CONFIG['hidden_size']}, "
          f"num_layers={Config.STUDENT_CONFIG['num_hidden_layers']}")
    print(f"⚙️ 训练参数: epochs={Config.TRAINING_ARGS['num_train_epochs']}, "
          f"batch_size={Config.TRAINING_ARGS['per_device_train_batch_size']}, "
          f"lr={Config.TRAINING_ARGS['learning_rate']}")
    print(f"🔢 精度模式: FP32")
    print("="*50 + "\n")
    
    try:
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            model=student_model,
            args=training_args,
            train_dataset=full_dataset,
            tokenizer=tokenizer
        )
        
        # 训练前打印模型参数数量
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        print(f"📊 模型参数: 教师模型={teacher_params:,} | 学生模型={student_params:,} "
              f"(压缩率: {teacher_params/student_params:.1f}x)")
        
        # 开始训练
        trainer.train()
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        raise
    
    # 5. 保存模型
    print("\n" + "="*50)
    print(f"💾 保存最终模型到: {Config.OUTPUT_DIR}")
    
    try:
        # 直接保存到主输出目录（与bert和tinybert训练保持一致）
        trainer.save_model(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)
        
        # 保存标签映射
        import json
        with open(os.path.join(Config.OUTPUT_DIR, "label_mapping.json"), "w") as f:
            json.dump({
                "id2label": Config.LABEL_MAPPING,
                "label2id": {v:k for k,v in Config.LABEL_MAPPING.items()}
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 蒸馏完成! 模型已保存至: {Config.OUTPUT_DIR}")
        print(f"总训练时间: {str(datetime.now() - trainer.start_time).split('.')[0]}")
        print(f"标签映射: {Config.LABEL_MAPPING}")
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
        raise

if __name__ == "__main__":
    main()
