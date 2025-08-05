######################################################
# 三合一模型转换与量化工具 (PyTorch → ONNX → INT8)
#
# 功能说明：
# - 完整流程：PyTorch模型 → ONNX转换 → 动态量化 → 精度验证
# - 支持独立执行每个步骤或完整流程
# - 提供交互式测试功能
#
# 主要功能模块：
# 1. 模型转换 (Conversion): 将PyTorch模型转换为ONNX格式
# 2. 模型量化 (Quantization): 将FP32 ONNX模型量化为INT8
# 3. 精度测试 (Accuracy Evaluation): 比较原始模型和量化模型的精度
#
# 需要修改的配置项（共9处需要修改，已在代码中用"x"标记）：
# 1. 原始PyTorch模型路径（1处需要修改）: 第38行
# 2. ONNX文件保存路径（1处需要修改）: 第41行
# 3. ONNX文件名（1处需要修改）: 第44行
# 4. 测试文本文件路径（1处需要修改，可选）: 第47行
# 5. 量化模型保存路径（1处需要修改）: 第50行
# 6. 测试数据集路径（1处需要修改）: 第53行
# 7. 标签映射（3处需要修改）：第62行、第155行、第220行
######################################################

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from safetensors.torch import load_file
import os
import numpy as np
from colorama import Fore, init
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import csv
from tqdm import tqdm

# ====================== 路径配置区 (直接修改这里) ======================
# 原始模型目录 (包含config.json和model.safetensors)
MODEL_DIR = os.path.expanduser("~/Desktop/try/trained_model_tinybert") # 修改为原始模型路径（1）

# ONNX输出目录 (确保此目录已存在)
ONNX_OUTPUT_DIR = os.path.expanduser("~/Desktop/try") # 修改为ONNX文件保存路径（2）

# ONNX文件名 (不需要带路径)
ONNX_FILENAME = "trained_model_tinybert.onnx" # 修改为ONNX文件名（2）

# 测试文件路径 (可选)
TEST_FILE_PATH = os.path.expanduser("~/Desktop/quantization/test.txt") # 修改测试文件（4）

# 量化模型路径
QUANT_MODEL_PATH = os.path.join(ONNX_OUTPUT_DIR, "trained_model_tinybert_quant.onnx") # 修改为量化后ONNX文件保存路径（5）

# 测试数据集路径
DATASET_PATH = os.path.expanduser("~/Desktop/train_conv_quant/dataset.csv") # 修改为测试集路径（6）

# 最大长度
MAX_LENGTH = 128  # Should match what you used during training
# ================================================================

init(autoreset=True)

def get_label_name(label_id):
    # 修改映射标签（7）
    labels = {
        0: "其他",
        1: "爱奇艺",
        2: "飞书",
        3: "鲁大师"
    }
    return labels.get(label_id, "未知")

class BertClassifier:
    def __init__(self):
        """初始化时自动使用预设路径"""
        self.model_dir = MODEL_DIR
        self.onnx_path = os.path.join(ONNX_OUTPUT_DIR, ONNX_FILENAME)
        self.load_models()
        
    def load_models(self):
        """加载模型"""
        print(f"{Fore.CYAN}\n=== 模型路径信息 ===")
        print(f"{Fore.YELLOW}PyTorch模型目录: {self.model_dir}")
        print(f"{Fore.YELLOW}ONNX输出路径: {self.onnx_path}")
        
        # 加载原始模型
        config = BertConfig.from_pretrained(self.model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        
        # PyTorch模型
        pt_model = BertForSequenceClassification(config)
        state_dict = load_file(os.path.join(self.model_dir, "model.safetensors"))
        pt_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()}, strict=False)
        pt_model.eval()
        self.pt_model = pt_model
        
        # 导出/加载ONNX
        if not os.path.exists(self.onnx_path):
            self.export_onnx()
        self.ort_session = ort.InferenceSession(self.onnx_path)
    
    def export_onnx(self):
        """导出ONNX模型"""
        print(f"{Fore.GREEN}\n正在导出ONNX到: {self.onnx_path}")
        sample_inputs = self.tokenizer("导出样例", return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        torch.onnx.export(
            self.pt_model,
            (sample_inputs["input_ids"], 
             sample_inputs["attention_mask"],
             sample_inputs["token_type_ids"]),
            self.onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            opset_version=14,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"}
            }
        )

    def predict(self, text, mode='both'):
        """预测函数"""
        inputs = self.tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }
        
        results = {}
        if mode in ['both', 'pytorch']:
            with torch.no_grad():
                outputs = self.pt_model(**{k: torch.tensor(v) for k, v in inputs.items()})
            results['pytorch'] = (np.argmax(outputs.logits.numpy()), torch.softmax(outputs.logits, dim=1).numpy()[0])
        
        if mode in ['both', 'onnx']:
            ort_outputs = self.ort_session.run(None, ort_inputs)
            results['onnx'] = (np.argmax(ort_outputs[0]), torch.softmax(torch.tensor(ort_outputs[0]), dim=1).numpy()[0])
        
        return results

    def interactive_test(self):
        """交互测试"""
        print(f"\n{Fore.CYAN}=== 交互模式 ===")
        print(f"{Fore.YELLOW}输入文本即时测试 (输入'quit'退出)")
        while True:
            text = input(f"\n{Fore.WHITE}测试文本: ")
            if text.lower() == 'quit':
                break
            self.display_results(text, self.predict(text))
    
    def batch_test(self):
        """批量测试"""

        # 修改映射标签（8）
        test_texts = [
            "爱奇艺视频非常好用",  
            "飞书是一款办公软件",  
            "鲁大师可以跑分",
            "今天天气真好"  
        ]
        
        # 如果存在测试文件则优先使用
        if os.path.exists(TEST_FILE_PATH):
            with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
                test_texts = [line.strip() for line in f if line.strip()]
            print(f"\n{Fore.GREEN}从文件加载测试用例: {TEST_FILE_PATH}")
        
        print(f"\n{Fore.CYAN}=== 批量测试 ===")
        for text in test_texts:
            self.display_results(text, self.predict(text))
    
    def display_results(self, text, results):
        """显示结果"""
        print(f"\n{Fore.YELLOW}文本: {Fore.WHITE}{text}")
        if 'pytorch' in results:
            pred, probs = results['pytorch']
            print(f"{Fore.GREEN}[PyTorch] 预测: {pred} ({get_label_name(pred)})")
        
        if 'onnx' in results:
            pred, probs = results['onnx']
            print(f"{Fore.MAGENTA}[ONNX] 预测: {pred} ({get_label_name(pred)})")

def quantize_onnx(model_path, quant_path):
    """执行动态量化"""
    if not os.path.exists(model_path):
        print(f"{Fore.RED}错误: ONNX模型文件不存在于 {model_path}")
        return False
    
    print(f"{Fore.GREEN}\n正在量化模型...")
    print(f"{Fore.YELLOW}输入模型: {model_path}")
    print(f"{Fore.YELLOW}输出模型: {quant_path}")
    
    try:
        # 动态量化（保持输出层精度）
        quantize_dynamic(
            model_path,
            quant_path,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=["logits"],  # 保持输出层为FP32
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": False}
        )
        
        print(f"{Fore.GREEN}✅ 量化完成！模型已保存到 {quant_path}")
        return True
    except Exception as e:
        print(f"{Fore.RED}量化过程中出错: {e}")
        return False

class ONNXModelTester:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        
        # Initialize ONNX runtime sessions
        self.sessions = {
            'original': ort.InferenceSession(os.path.join(ONNX_OUTPUT_DIR, ONNX_FILENAME)),
            'quantized': ort.InferenceSession(QUANT_MODEL_PATH)
        }
        
        # 修改映射标签（9）
        self.label_map = {
            0: "其他",
            1: "爱奇艺",
            2: "飞书",
            3: "鲁大师"
        }

    def preprocess(self, text):
        """Tokenize and prepare input for ONNX models"""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True
        )
        return {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }

    def predict(self, text, model_type='original'):
        """Make prediction with specified model"""
        inputs = self.preprocess(text)
        outputs = self.sessions[model_type].run(None, inputs)
        logits = outputs[0]
        pred = np.argmax(logits, axis=1)[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return pred, probs[0]

    def evaluate_dataset(self):
        """Evaluate both models on the entire dataset with progress bar"""
        if not os.path.exists(DATASET_PATH):
            print(f"{Fore.RED}Dataset not found at {DATASET_PATH}")
            return

        # First count total rows for progress bar
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in csv.DictReader(f))

        if total_rows == 0:
            print(f"{Fore.RED}No samples found in dataset")
            return

        correct_counts = {'original': 0, 'quantized': 0}
        processed = 0
        errors = 0

        print(f"\n{Fore.CYAN}Evaluating on {total_rows} samples...")
        
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, desc="Processing", unit="samples"):
                try:
                    true_label = int(row['label'])
                    text = row['text']
                    processed += 1

                    for model_type in ['original', 'quantized']:
                        pred, _ = self.predict(text, model_type)
                        if pred == true_label:
                            correct_counts[model_type] += 1

                except Exception as e:
                    errors += 1
                    continue

        print(f"\n{Fore.CYAN}=== Evaluation Results ===")
        print(f"{Fore.WHITE}Total samples: {processed} (Errors: {errors})")
        
        for model_type in ['original', 'quantized']:
            accuracy = correct_counts[model_type] / processed * 100
            color = Fore.GREEN if model_type == 'original' else Fore.YELLOW
            print(f"{color}{model_type.capitalize()} model accuracy: {accuracy:.2f}%")
            print(f"  Correct: {correct_counts[model_type]}/{processed}")

    def interactive_test(self):
        """Interactive testing mode"""
        print(f"\n{Fore.CYAN}=== Interactive Testing ===")
        print(f"{Fore.YELLOW}Enter text to test (type 'exit' to quit)")
        
        while True:
            text = input(f"\n{Fore.WHITE}Input text: ").strip()
            if text.lower() == 'exit':
                break
            
            if not text:
                continue
            
            print(f"\n{Fore.BLUE}Testing: {text}")
            
            for model_type in ['original', 'quantized']:
                pred, probs = self.predict(text, model_type)
                print(f"\n{Fore.MAGENTA}{model_type.capitalize()} Model:")
                print(f"{Fore.GREEN}Predicted: {pred} ({self.label_map.get(pred, 'Unknown')})")
                print(f"{Fore.CYAN}Probabilities:")
                for i, prob in enumerate(probs):
                    print(f"  {self.label_map.get(i, i)}: {prob:.4f}")

def run_conversion():
    """运行模型转换流程"""
    print(f"\n{Fore.GREEN}=== 开始模型转换 ===")
    classifier = BertClassifier()
    print(f"{Fore.GREEN}模型转换完成!")
    return classifier

def run_quantization():
    """运行模型量化流程"""
    print(f"\n{Fore.GREEN}=== 开始模型量化 ===")
    onnx_path = os.path.join(ONNX_OUTPUT_DIR, ONNX_FILENAME)
    quant_path = QUANT_MODEL_PATH
    
    if not os.path.exists(onnx_path):
        print(f"{Fore.RED}错误: ONNX模型文件不存在，请先运行模型转换")
        return False
    
    return quantize_onnx(onnx_path, quant_path)

def run_accuracy_test():
    """运行精度测试流程"""
    print(f"\n{Fore.GREEN}=== 开始精度测试 ===")
    tester = ONNXModelTester()
    
    # 检查量化模型是否存在
    if not os.path.exists(QUANT_MODEL_PATH):
        print(f"{Fore.YELLOW}警告: 量化模型不存在，将只测试原始ONNX模型")
        tester.sessions.pop('quantized', None)
    
    tester.evaluate_dataset()
    return True

def main():
    """主函数"""
    init(autoreset=True)
    
    while True:
        print(f"\n{Fore.CYAN}=== 主菜单 ===")
        print(f"{Fore.YELLOW}1. 模型转换 (Conversion)")
        print(f"{Fore.YELLOW}2. 模型量化 (Quantization)")
        print(f"{Fore.YELLOW}3. 精度测试 (Accuracy Evaluation)")
        print(f"{Fore.YELLOW}4. 完整流程 (Conversion → Quantization → Accuracy)")
        print(f"{Fore.YELLOW}5. 退出")
        
        choice = input(f"{Fore.WHITE}选择操作 (1-5): ")
        
        if choice == '1':
            # 模型转换流程
            classifier = run_conversion()
            
            # 提供测试选项
            test_choice = input(f"{Fore.WHITE}是否要测试转换后的模型? (y/n): ").lower()
            if test_choice == 'y':
                print(f"\n{Fore.CYAN}=== 测试选项 ===")
                print(f"{Fore.YELLOW}1. 交互测试")
                print(f"{Fore.YELLOW}2. 批量测试")
                test_mode = input(f"{Fore.WHITE}选择测试模式 (1-2): ")
                
                if test_mode == '1':
                    classifier.interactive_test()
                elif test_mode == '2':
                    classifier.batch_test()
                    
        elif choice == '2':
            # 模型量化流程
            if run_quantization():
                # 量化成功后提供测试选项
                test_choice = input(f"{Fore.WHITE}是否要测试量化后的模型? (y/n): ").lower()
                if test_choice == 'y':
                    tester = ONNXModelTester()
                    tester.interactive_test()
                    
        elif choice == '3':
            # 精度测试流程
            run_accuracy_test()
            
        elif choice == '4':
            # 完整流程
            print(f"\n{Fore.GREEN}=== 开始完整流程 ===")
            
            # 1. 模型转换
            print(f"\n{Fore.CYAN}>>> 步骤1: 模型转换 <<<")
            classifier = run_conversion()
            
            # 2. 模型量化
            print(f"\n{Fore.CYAN}>>> 步骤2: 模型量化 <<<")
            if run_quantization():
                # 3. 精度测试
                print(f"\n{Fore.CYAN}>>> 步骤3: 精度测试 <<<")
                run_accuracy_test()
            
        elif choice == '5':
            print(f"{Fore.GREEN}退出程序...")
            break
            
        else:
            print(f"{Fore.RED}无效输入，请重新选择")

if __name__ == "__main__":
    main()