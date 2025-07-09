################################################
# 本文件用于检查onnx格式下模型的推理时间和内存
################################################

import os
import time
import psutil
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

def get_memory_usage():
    """获取当前进程内存使用量 (MB) / Get current process memory usage (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # 转换为MB / Convert to MB

def test_model(model_path, model_name):
    """测试单个模型的性能和内存 / Test performance and memory for a single model"""
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件未找到: {model_path} / [Error] Model file not found: {model_path}")
        return None

    print(f"\n=== 测试 {model_name} 模型 === / === Testing {model_name} model ===")
    
    # 内存基准 / Memory baseline
    mem_before = get_memory_usage()
    
    # 加载模型 / Load model
    load_start = time.time()
    session = ort.InferenceSession(model_path)
    load_time = time.time() - load_start
    mem_after_load = get_memory_usage()
    
    # 准备输入 / Prepare input
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    inputs = tokenizer("测试模型性能", return_tensors="np")
    inputs = {k: v.astype(np.int64) for k,v in inputs.items()}
    
    # 预热 / Warm-up
    for _ in range(3):
        session.run(None, inputs)
    
    # 正式推理测试 / Formal inference test
    inference_times = []
    for _ in range(10):  # 运行10次取平均 / Run 10 times for average
        start_time = time.time()
        outputs = session.run(None, inputs)
        inference_times.append(time.time() - start_time)
    
    avg_inference = np.mean(inference_times) * 1000  # 转换为毫秒 / Convert to milliseconds
    mem_after_infer = get_memory_usage()
    
    # 打印结果 / Print results
    print(f"1. 加载时间 (Load time): {load_time:.3f}s")
    print(f"2. 平均推理时间 (Avg inference): {avg_inference:.2f}ms")
    print(f"3. 内存变化 (Memory change):")
    print(f"   - 加载前 (Before load): {mem_before:.2f}MB")
    print(f"   - 加载后 (After load): {mem_after_load:.2f}MB (+{mem_after_load - mem_before:.2f}MB)")
    print(f"   - 推理后 (After inference): {mem_after_infer:.2f}MB")
    print(f"4. 模型大小 (Model size): {os.path.getsize(model_path)/1024/1024:.2f}MB")
    
    return {
        'load_time': load_time,
        'avg_inference': avg_inference,
        'mem_usage': mem_after_infer,
        'model_size': os.path.getsize(model_path)
    }

def main():
    # 模型配置 / Model configuration
    models = {
        'original': 'bert_model.onnx',  # 原始模型 / Original model
        'quantized': 'bert_model_quant.onnx'  # 量化模型 / Quantized model
    }
    
    print("=== ONNX模型性能测试工具 ===")
    print("(ONNX Model Performance Testing Tool)")
    
    results = {}
    for name, path in models.items():
        results[name] = test_model(path, name)
    
    # 比较结果 / Compare results
    if all(results.values()):
        print("\n=== 模型比较结果 ===")
        print("(Model Comparison Results)")
        
        orig = results['original']
        quant = results['quantized']
        
        print(f"1. 推理速度提升 (Inference speed improvement): "
              f"{orig['avg_inference']/quant['avg_inference']:.1f}x faster")
        print(f"2. 内存占用减少 (Memory reduction): "
              f"{orig['mem_usage'] - quant['mem_usage']:.2f}MB ({(1-quant['mem_usage']/orig['mem_usage'])*100:.1f}%)")
        print(f"3. 模型大小减少 (Model size reduction): "
              f"{(orig['model_size'] - quant['model_size'])/1024/1024:.2f}MB ({(1-quant['model_size']/orig['model_size'])*100:.1f}%)")

if __name__ == "__main__":
    main()