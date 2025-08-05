######################################################
# 比较两个safetensors文件是否相同
#
# 功能说明：
# - 输入: 两个safetensors文件
# - 输出: 判断二者是否相同
#
# 需要修改的配置项（共2处需要修改，已在代码中用注释"x"标记）：
# 1. 第一个safetensors文件路径: 第63行
# 2. 第二个safetensors文件路径: 第64行
######################################################


import torch
from safetensors import safe_open

def compare_safetensors(file1, file2, rtol=1e-5, atol=1e-8):
    """
    Compare two safetensors files and check if their tensors are equal within tolerances.
    
    Args:
        file1 (str): Path to first safetensors file
        file2 (str): Path to second safetensors file
        rtol (float): Relative tolerance
        atol (float): Absolute tolerance
    """
    # Open both files
    with safe_open(file1, framework="pt") as f1, safe_open(file2, framework="pt") as f2:
        # Get all tensor names
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        
        # Check if keys match
        if keys1 != keys2:
            print("Keys don't match!")
            print(f"Keys in {file1} but not {file2}: {keys1 - keys2}")
            print(f"Keys in {file2} but not {file1}: {keys2 - keys1}")
            return False
        
        # Compare each tensor
        all_equal = True
        for key in keys1:
            tensor1 = f1.get_tensor(key)
            tensor2 = f2.get_tensor(key)
            
            if tensor1.shape != tensor2.shape:
                print(f"Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
                all_equal = False
                continue
                
            if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
                print(f"Values differ for {key}")
                all_equal = False
        
        if all_equal:
            print("All tensors match within tolerances!")
            return True
        else:
            return False

# Example usage
if __name__ == "__main__":
    file1 = "tinybert_split_p/model.safetensors" # 改为第一个safetensors文件路径（1）
    file2 = "tinybert_split/model.safetensors" # 改为第二个safetensors文件路径（2）
    compare_safetensors(file1, file2)