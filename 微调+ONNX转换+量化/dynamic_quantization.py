######################################################
# ONNX 模型动态量化脚本 (FP32 → INT8)
#
# 功能说明：
# - 输入: FP32精度的原始ONNX模型
# - 输出: INT8精度的量化ONNX模型
#
# 需要修改的配置项（共3处需要修改，已在代码中用注释"x"标记）：
# 1. 原始模型路径（1处需要修改）: 第36行
# 2. 量化模型保存路径（2处需要修改）: 第17行、第37行
######################################################

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os

def quantize_onnx(model_path, quant_path="trained_model_bert_full_data_128_quant.onnx"): # 改为量化后的ONNX模型的保存路径（1）
    """执行动态量化"""
    # 加载原始模型
    model = onnx.load(model_path)
    
    # 动态量化（保持输出层精度）
    quantize_dynamic(
        model_path,
        quant_path,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=["logits"],  # 保持输出层为FP32
        extra_options={"WeightSymmetric": False, "ActivationSymmetric": False}
    )
    
    print(f"✅ 量化完成！模型已保存到 {os.path.abspath(quant_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="trained_model_bert_full_data_128.onnx") # 改为原始ONNX模型的路径（2）
    parser.add_argument("--output", default="trained_model_bert_full_data_128_quant.onnx") # 改为量化后的ONNX模型的保存路径（3）
    args = parser.parse_args()
    
    quantize_onnx(args.input, args.output)