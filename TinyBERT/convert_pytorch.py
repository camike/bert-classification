from safetensors import safe_open
import torch

# 输入和输出路径
model_dir = "../trained_model/model/"
input_path = model_dir + "/model.safetensors"  # 输入文件
output_path = model_dir + "/pytorch_model.bin"         # 输出文件

# 从 .safetensors 加载权重
tensors = {}
with safe_open(input_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# 保存为 PyTorch .bin 格式
torch.save(tensors, output_path)
print(f"转换完成！保存至: {output_path}")