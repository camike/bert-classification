import os
model_path = "tinybert_model.onnx"
improved_model_path = "tinybert_model_quant.onnx"
print(f"原始模型大小: {os.path.getsize(model_path)/1024/1024:.2f}MB")  # 应显示400MB左右
print(f"量化模型大小: {os.path.getsize(improved_model_path)/1024/1024:.2f}MB")  # 应显示400MB左右