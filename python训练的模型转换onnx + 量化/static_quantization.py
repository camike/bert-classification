from onnxruntime.quantization import (
    quantize_static, 
    QuantType, 
    CalibrationDataReader,
    quant_pre_process
)
from transformers import AutoTokenizer
import numpy as np
import csv
import os
from tqdm import tqdm
import time

# ====== 配置参数 ======
INPUT_MODEL_PATH = "bert_model.onnx"
OUTPUT_MODEL_PATH = "bert_model_quant.onnx"
CALIBRATION_CSV_PATH = "dataset.csv"
# =====================

class ProgressCalibrationDataReader(CalibrationDataReader):
    def __init__(self, csv_path, tokenizer_name, input_names):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_names = input_names
        
        # 先计算总行数
        with open(csv_path, "r", encoding='utf-8') as f:
            self.total_samples = sum(1 for _ in csv.reader(f)) - 1
        
        # 读取数据
        self.data = []
        with open(csv_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            text_index = header.index("text")
            for row in tqdm(reader, total=self.total_samples, desc="📊 加载校准数据"):
                text = row[text_index]
                encoded = self.tokenizer(
                    text,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )
                input_dict = {
                    "input_ids": encoded["input_ids"].astype(np.int64),
                    "attention_mask": encoded["attention_mask"].astype(np.int64),
                    "token_type_ids": np.zeros_like(encoded["input_ids"]).astype(np.int64)
                }
                self.data.append(input_dict)
        
        self.current_idx = 0
        self.pbar = tqdm(total=self.total_samples, desc="🔧 校准处理")

    def get_next(self):
        if self.current_idx >= len(self.data):
            self.pbar.close()
            return None
        item = self.data[self.current_idx]
        self.current_idx += 1
        self.pbar.update(1)
        return item

def preprocess_model():
    print("🔄 正在预处理模型...")
    preprocessed_path = INPUT_MODEL_PATH.replace(".onnx", "_preprocessed.onnx")
    quant_pre_process(
        input_model_path=INPUT_MODEL_PATH,
        output_model_path=preprocessed_path,
        auto_merge=True
    )
    print(f"✅ 预处理完成: {preprocessed_path}")
    return preprocessed_path

def run_quantization():
    # 检查文件
    if not all(os.path.exists(p) for p in [INPUT_MODEL_PATH, CALIBRATION_CSV_PATH]):
        raise FileNotFoundError("输入文件不存在")
    
    # 1. 预处理
    processed_model = preprocess_model()
    
    # 2. 量化
    print("⏳ 开始量化处理...")
    start_time = time.time()
    
    quantize_static(
        model_input=processed_model,
        model_output=OUTPUT_MODEL_PATH,
        calibration_data_reader=ProgressCalibrationDataReader(
            csv_path=CALIBRATION_CSV_PATH,
            tokenizer_name="bert-base-chinese",
            input_names=["input_ids", "attention_mask", "token_type_ids"]
        ),
        quant_format="QDQ",
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False
        }
    )
    
    # 清理并输出结果
    os.remove(processed_model)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f"\n✅ 量化完成! 用时: {elapsed}")
    print(f"输出模型: {os.path.abspath(OUTPUT_MODEL_PATH)}")

if __name__ == "__main__":
    try:
        run_quantization()
    except Exception as e:
        print(f"❌ 错误: {str(e)}")