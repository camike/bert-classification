################################################
# 本文件用于检查模型推理的准确率
#
# 功能说明：
# - 输入: 原始模型路径；ONNX量化前后路径模型；测试集
# - 输出: 量化前后模型在测试集上的准确率
#
# 需要修改的配置项（共4处需要修改，已在代码中用注释"x"标记）：第25-28行
################################################

import os
import csv
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer
from colorama import Fore, Style, init
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

# ====================== CONFIGURATION ======================
MODEL_DIR = os.path.expanduser("~/Desktop/quantization/bert_teacher") # 改为原始模型文件夹路径（1）
ONNX_MODEL_PATH = os.path.expanduser("~/Desktop/quantization/bert_teacher.onnx") # 改为量化前ONNX文件路径（2）
QUANT_MODEL_PATH = os.path.expanduser("~/Desktop/quantization/bert_teacher_quant.onnx") # 改为量化后ONNX文件路径（3）
DATASET_PATH = os.path.expanduser("~/Desktop/quantization/dataset.csv") # 改为测试集路径（4）

MAX_LENGTH = 128  # Should match what you used during training
# ===========================================================

class ONNXModelTester:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        
        # Initialize ONNX runtime sessions
        self.sessions = {
            'original': ort.InferenceSession(ONNX_MODEL_PATH),
            'quantized': ort.InferenceSession(QUANT_MODEL_PATH)
        }
        
        # Label mapping (update with your actual labels)
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
            
            # Show correct/total counts
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

def main():
    tester = ONNXModelTester()
    
    while True:
        print(f"\n{Fore.CYAN}=== Main Menu ===")
        print(f"{Fore.YELLOW}1. Evaluate on full dataset")
        print(f"{Fore.YELLOW}2. Interactive testing")
        print(f"{Fore.YELLOW}3. Exit")
        
        choice = input(f"{Fore.WHITE}Select option (1-3): ")
        
        if choice == '1':
            tester.evaluate_dataset()
        elif choice == '2':
            tester.interactive_test()
        elif choice == '3':
            break
        else:
            print(f"{Fore.RED}Invalid choice")

if __name__ == "__main__":
    main()