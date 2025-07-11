import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from safetensors.torch import load_file
import os
import numpy as np
from colorama import Fore, init
import onnxruntime as ort

# ====================== 路径配置区 (直接修改这里) ======================
# 原始模型目录 (包含config.json和model.safetensors)
MODEL_DIR = os.path.expanduser("~/Desktop/quantization/trained_model_bert")

# ONNX输出目录 (确保此目录已存在)
ONNX_OUTPUT_DIR = os.path.expanduser("~/Desktop/quantization")

# ONNX文件名 (不需要带路径)
ONNX_FILENAME = "bert_512.onnx"

# 测试文件路径 (可选)
TEST_FILE_PATH = os.path.expanduser("~/Desktop/quantization/test_samples.txt")
# ================================================================

init(autoreset=True)

def get_label_name(label_id):
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
        test_texts = [
            "爱奇艺视频非常好用",  # 应预测为1
            "飞书是一款办公软件",  # 应预测为2
            "鲁大师可以跑分",     # 应预测为3
            "今天天气真好"       # 应预测为0
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
            # print(f"{Fore.BLUE}概率: {', '.join([f'{p:.4f}' for p in probs])}")
        
        if 'onnx' in results:
            pred, probs = results['onnx']
            print(f"{Fore.MAGENTA}[ONNX] 预测: {pred} ({get_label_name(pred)})")
            # print(f"{Fore.CYAN}概率: {', '.join([f'{p:.4f}' for p in probs])}")

def main():
    classifier = BertClassifier()
    
    while True:
        print(f"\n{Fore.CYAN}=== 操作菜单 ===")
        print(f"{Fore.YELLOW}1. 交互测试")
        print(f"{Fore.YELLOW}2. 批量测试")
        print(f"{Fore.YELLOW}3. 重新导出ONNX")
        print(f"{Fore.YELLOW}4. 退出")
        
        choice = input(f"{Fore.WHITE}选择操作 (1-4): ")
        
        if choice == '1':
            classifier.interactive_test()
        elif choice == '2':
            classifier.batch_test()
        elif choice == '3':
            # Only delete if file exists
            if os.path.exists(classifier.onnx_path):
                os.remove(classifier.onnx_path)
            classifier.export_onnx()  # This will create new ONNX file
            print(f"{Fore.GREEN}ONNX模型已重新导出!")
        elif choice == '4':
            break
        else:
            print(f"{Fore.RED}无效输入")

if __name__ == "__main__":
    main()