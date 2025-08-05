################################################
# 本文件用于微调TinyBert模型（未设置验证集）
#
# 功能说明：
# - 输入: 模型保存路径；训练集
# - 输出: 微调后的TinyBert模型
#
# 需要修改的配置项（共4处需要修改，已在代码中用注释"x"标记）：
# 1. 训练集路径: 第84行
# 2. 标签映射: 第100行
# 3. test trainer文件保存路径: 第150行
# 4. 微调模型文件保存路径: 第190行 
################################################

from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import random
import os
import shutil
from datetime import datetime

# 设置Hugging Face镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置cuBLAS确定性环境变量
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 设置全局随机种子以确保复现性
seed = 521
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 增强确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# 尝试使用确定性算法
try:
    torch.use_deterministic_algorithms(True)
    print("已启用确定性算法")
except Exception as e:
    print(f"警告: 无法启用确定性算法: {str(e)}")

now = datetime.now()
print("start time", now)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "huawei-noah/TinyBERT_4L_zh"  # 使用华为的中文TinyBERT

# 创建固定的tokenizer目录
fixed_tokenizer_dir = "./fixed_tokenizer_tinybert"

# 检查是否已经存在固定的tokenizer
if not os.path.exists(fixed_tokenizer_dir):
    print("创建固定的tokenizer...")
    os.makedirs(fixed_tokenizer_dir, exist_ok=True)

    # 加载原始tokenizer
    original_tokenizer = BertTokenizer.from_pretrained(model_path)

    # 保存到固定位置
    original_tokenizer.save_pretrained(fixed_tokenizer_dir)
    print(f"固定tokenizer已保存到 {fixed_tokenizer_dir}")
else:
    print(f"使用已存在的固定tokenizer: {fixed_tokenizer_dir}")

# 加载固定的tokenizer
tokenizer = BertTokenizer.from_pretrained(fixed_tokenizer_dir)
print("成功加载固定tokenizer")

# 加载所有数据作为训练集
dataset = load_dataset("csv", data_files="dataset.csv", split="train") # 修改为训练集路径（1）
print(f"数据集大小: {len(dataset)}")

# 使用固定的tokenizer处理数据集
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    ), 
    batched=True
)

print(f"处理后的数据集大小: {len(tokenized_dataset)}")

# 标签映射按需修改（2）
id2label = {
    0: "其他",
    1: "爱奇艺",
    2: "飞书",
    3: "鲁大师",
}

label2id = {v: k for k, v in id2label.items()}

model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=4,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
   
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
   
    try:
        logits_tensor = torch.tensor(logits)
        import torch.nn.functional as F
        probs = F.softmax(logits_tensor, dim=1).numpy()
        auc_roc = roc_auc_score(
            labels,
            probs,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        auc_roc = float('nan')
   
    return {
        "accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

# 创建输出目录
output_dir = "test_trainer_tinybert" # 修改为test trainer文件保存路径（3）
if os.path.exists(output_dir):
    print(f"清理旧的输出目录: {output_dir}")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=8,  # 增加训练轮数
    learning_rate=2e-5,  # 提高学习率
    per_device_train_batch_size=16,  # 优化批次大小
    gradient_accumulation_steps=2,   # 添加梯度累积
    save_strategy="epoch",
    eval_strategy="no",
    save_total_limit=1,
    seed=seed,
    data_seed=seed,
    logging_steps=10000,
    disable_tqdm=False,
    report_to="none",
    warmup_steps=200,  # 添加warmup
    weight_decay=0.01,  # 添加权重衰减
    lr_scheduler_type="cosine",  # 使用余弦学习率调度
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    compute_metrics=compute_metrics
)

# 训练模型
print("\n开始训练模型...")
trainer.train()

now = datetime.now()
print("end time", now)

# 确保模型输出目录存在
model_output_dir = "trained_model_tinybert" # 修改为微调模型文件保存路径（4）
if os.path.exists(model_output_dir):
    print(f"清理旧的模型目录: {model_output_dir}")
    shutil.rmtree(model_output_dir)
os.makedirs(model_output_dir, exist_ok=True)

# 保存模型
trainer.model.save_pretrained(model_output_dir)
# 复制固定的tokenizer到模型目录
print(f"复制固定tokenizer到模型目录: {model_output_dir}")
for file_name in os.listdir(fixed_tokenizer_dir):
    source_file = os.path.join(fixed_tokenizer_dir, file_name)
    target_file = os.path.join(model_output_dir, file_name)
    shutil.copy2(source_file, target_file)

print(f"模型和tokenizer已保存到: {model_output_dir}")

# 使用训练后的模型对训练集进行预测并计算正确率
print("\n========== 训练集评估 ==========")
print("使用最终训练后的模型对训练集进行预测...")

try:
    # 对训练集进行预测
    train_predictions = trainer.predict(tokenized_dataset)
    
    # 确保predictions不为None并且有predictions属性
    if train_predictions is not None and hasattr(train_predictions, 'predictions'):
        train_preds = np.argmax(train_predictions.predictions, axis=-1)
        train_labels = train_predictions.label_ids
        train_accuracy = (train_preds == train_labels).mean()

        print(f"训练集上的整体准确率: {train_accuracy:.4f}")

        # 计算每个类别的准确率
        class_metrics = {}
        for class_id, class_name in id2label.items():
            class_indices = (train_labels == class_id)
            if np.sum(class_indices) > 0:  # 确保该类别有样本
                class_accuracy = np.mean(train_preds[class_indices] == train_labels[class_indices])
                class_count = np.sum(class_indices)
                print(f"类别 '{class_name}' (ID: {class_id}) 的准确率: {class_accuracy:.4f}, 样本数: {class_count}")
                
                # 计算该类别的精确率、召回率和F1值
                true_positives = np.sum((train_preds == class_id) & (train_labels == class_id))
                false_positives = np.sum((train_preds == class_id) & (train_labels != class_id))
                false_negatives = np.sum((train_preds != class_id) & (train_labels == class_id))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}")
                
                class_metrics[class_name] = {
                    "accuracy": class_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sample_count": int(class_count)
                }

        # 计算整体指标
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average="macro"
        )

        print("\n整体评估指标:")
        print(f"准确率: {train_accuracy:.4f}")
        print(f"宏平均精确率: {overall_precision:.4f}")
        print(f"宏平均召回率: {overall_recall:.4f}")
        print(f"宏平均F1值: {overall_f1:.4f}")

        # 计算混淆矩阵
        print("\n混淆矩阵:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(train_labels, train_preds)
        print(cm)
    else:
        print("警告: 预测结果格式不符合预期，可能是预测过程中出现了错误。")
except Exception as e:
    print(f"评估过程中出现错误: {str(e)}")

print("\n训练和评估完成！")
