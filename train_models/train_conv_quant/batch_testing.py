################################################
# 本文件用于导入测试集检查模型推理的准确率
#
# 功能说明：
# - 输入: 原始模型路径；ONNX量化前后路径模型；测试集
# - 输出: 量化前后模型在测试集上的准确率
################################################

import os
import csv
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import json
from colorama import Fore, Style, init

# 初始化colorama
init(autoreset=True)

# ====================== CONFIGURATION ======================
MODEL_DIR = "trained_model_tinybert" # 改为原始模型文件夹路径（1）
ONNX_MODEL_PATH = "trained_model_tinybert.onnx" # 改为量化前ONNX文件路径（2）
QUANT_MODEL_PATH = "trained_model_tinybert_quant.onnx" # 改为量化后ONNX文件路径（3）
DATASET_PATH = "test_dataset.csv" # 测试数据集路径（4）

MAX_LENGTH = 128  # Should match what you used during training
# ===========================================================

class DatasetTester:
    def __init__(self):
        self.tokenizer = None
        self.sessions = {}
        self.label_map = {
            0: "其他",
            1: "爱奇艺", 
            2: "飞书",
            3: "鲁大师"
        }
        self.load_models()
    
    def load_models(self):
        """Load tokenizer and ONNX models"""
        try:
            # Load tokenizer
            if os.path.exists(MODEL_DIR):
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                print(f"{Fore.GREEN}✅ Tokenizer loaded from {MODEL_DIR}")
            else:
                print(f"{Fore.RED}❌ Model directory not found: {MODEL_DIR}")
                return
            
            # Load original ONNX model
            if os.path.exists(ONNX_MODEL_PATH):
                self.sessions['original'] = ort.InferenceSession(ONNX_MODEL_PATH)
                print(f"{Fore.GREEN}✅ Original ONNX model loaded: {ONNX_MODEL_PATH}")
            else:
                print(f"{Fore.YELLOW}⚠️ Original ONNX model not found: {ONNX_MODEL_PATH}")
            
            # Load quantized ONNX model
            if os.path.exists(QUANT_MODEL_PATH):
                self.sessions['quantized'] = ort.InferenceSession(QUANT_MODEL_PATH)
                print(f"{Fore.GREEN}✅ Quantized ONNX model loaded: {QUANT_MODEL_PATH}")
            else:
                print(f"{Fore.YELLOW}⚠️ Quantized ONNX model not found: {QUANT_MODEL_PATH}")
            
            if not self.sessions:
                raise Exception("No ONNX models found!")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to load models: {e}")
            raise e
    
    def preprocess(self, text):
        """Preprocess text for model input"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='np'
        )
        
        # Ensure all required inputs are present
        inputs = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64)
        }
        
        # Add token_type_ids if available in encoding
        if 'token_type_ids' in encoding:
            inputs['token_type_ids'] = encoding['token_type_ids'].astype(np.int64)
        else:
            # Create token_type_ids manually (all zeros for single sentence)
            inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'], dtype=np.int64)
        
        return inputs
    
    def predict(self, text, model_type='original'):
        """Make prediction with specified model"""
        inputs = self.preprocess(text)
        outputs = self.sessions[model_type].run(None, inputs)
        logits = outputs[0]
        pred = np.argmax(logits, axis=1)[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return pred, probs[0]
    
    def load_test_dataset(self, dataset_path):
        """Load test dataset from CSV file"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        texts = []
        labels = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    label = int(row['label'])
                    text = row['text'].strip()
                    if text and label in self.label_map:
                        texts.append(text)
                        labels.append(label)
                except (ValueError, KeyError) as e:
                    print(f"{Fore.YELLOW}⚠️ Skipping invalid row: {row}")
                    continue
        
        print(f"{Fore.GREEN}✅ Loaded {len(texts)} samples from dataset")
        return texts, labels
    
    def evaluate_dataset(self, dataset_path=None, return_errors=True, progress_callback=None):
        """Evaluate models on test dataset"""
        if dataset_path is None:
            dataset_path = DATASET_PATH
            
        # Load test dataset
        texts, true_labels = self.load_test_dataset(dataset_path)
        
        if len(texts) == 0:
            return {"error": "No valid samples found in dataset"}
        
        results = {}
        
        # Test each model
        for model_idx, model_type in enumerate(['original', 'quantized']):
            if model_type not in self.sessions:
                continue
                
            print(f"\n{Fore.CYAN}=== Testing {model_type.capitalize()} Model ===")
            
            predictions = []
            probabilities = []
            errors = []
            
            # Update progress
            if progress_callback:
                progress_callback(60 + model_idx * 20, f'测试{model_type}模型...')
            
            # Make predictions with progress bar
            for i, text in enumerate(tqdm(texts, desc=f"Testing {model_type}", unit="samples")):
                try:
                    pred, probs = self.predict(text, model_type)
                    predictions.append(pred)
                    probabilities.append(probs)
                    
                    # Record errors if prediction is wrong
                    if return_errors and pred != true_labels[i]:
                        errors.append({
                            'text': text,
                            'true_label': true_labels[i],
                            'true_label_name': self.label_map[true_labels[i]],
                            'predicted_label': pred,
                            'predicted_label_name': self.label_map[pred],
                            'confidence': float(probs[pred])
                        })
                        
                    # Update progress periodically
                    if progress_callback and i % 10 == 0:
                        progress = 60 + model_idx * 20 + (i / len(texts)) * 20
                        progress_callback(int(progress), f'测试{model_type}模型: {i+1}/{len(texts)}')
                        
                except Exception as e:
                    print(f"{Fore.RED}❌ Error processing sample {i}: {e}")
                    continue
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='macro', zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)
            
            # Classification report
            report = classification_report(
                true_labels, predictions, 
                target_names=[self.label_map[i] for i in sorted(self.label_map.keys())],
                output_dict=True
            )
            
            # Store results
            results[model_type] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'total_samples': len(texts),
                'errors': errors[:50] if return_errors else []  # Limit to first 50 errors
            }
            
            # Print results
            print(f"\n{Fore.GREEN}整体评估指标:")
            print(f"{Fore.WHITE}准确率: {accuracy:.4f}")
            print(f"{Fore.WHITE}宏平均精确率: {precision:.4f}")
            print(f"{Fore.WHITE}宏平均召回率: {recall:.4f}")
            print(f"{Fore.WHITE}宏平均F1值: {f1:.4f}")
            print(f"\n{Fore.GREEN}混淆矩阵:")
            print(f"{Fore.WHITE}{cm}")
            
            if errors:
                print(f"\n{Fore.YELLOW}【{model_type.upper()}模型】错误样本数量: {len(errors)}")
                print(f"{Fore.YELLOW}显示前10个错误样本:")
                for i, error in enumerate(errors[:10]):
                    print(f"{Fore.RED}  {i+1}. 文本: {error['text'][:80]}...")
                    print(f"{Fore.RED}     真实: {error['true_label_name']} | 预测: {error['predicted_label_name']} | 置信度: {error['confidence']:.3f}")
                    print()
        
        # Print comparison summary if both models were tested
        if len(results) == 2:
            print(f"\n{Fore.CYAN}=== 模型对比总结 ===")
            original_acc = results.get('original', {}).get('accuracy', 0)
            quantized_acc = results.get('quantized', {}).get('accuracy', 0)
            print(f"{Fore.WHITE}原始模型准确率: {original_acc:.4f}")
            print(f"{Fore.WHITE}量化模型准确率: {quantized_acc:.4f}")
            print(f"{Fore.WHITE}准确率差异: {abs(original_acc - quantized_acc):.4f}")
            
            # Compare error counts
            original_errors = len(results.get('original', {}).get('errors', []))
            quantized_errors = len(results.get('quantized', {}).get('errors', []))
            print(f"{Fore.WHITE}原始模型错误数: {original_errors}")
            print(f"{Fore.WHITE}量化模型错误数: {quantized_errors}")
        
        return results
    
    def save_results(self, results, output_path="evaluation_results.json"):
        """Save evaluation results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"{Fore.GREEN}✅ Results saved to {output_path}")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to save results: {e}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX models on dataset")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to test dataset CSV file")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        tester = DatasetTester()
        results = tester.evaluate_dataset(args.dataset)
        tester.save_results(results, args.output)
        
        print(f"\n{Fore.GREEN}🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

