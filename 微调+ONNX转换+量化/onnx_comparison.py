######################################################
# 比较两个ONNX文件是否相同
#
# 功能说明：
# - 输入: 两个ONNX文件路径
# - 输出: 判断二者是否相同
#
# 需要修改的配置项（共2处需要修改，已在代码中用注释"x"标记）：
# 1. 第一个ONNX文件路径（1处需要修改）: 第52行
# 2. 第二个ONNX文件路径（1处需要修改）: 第53行
######################################################


from onnxruntime.quantization import quantize_dynamic, QuantType
import hashlib
import os

def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def compare_onnx_files(file1_path, file2_path):
    """Compare two ONNX files by hash"""
    print(f"\nComparing files:\n1. {file1_path}\n2. {file2_path}")
    
    if not os.path.exists(file1_path):
        print(f"❌ File not found: {file1_path}")
        return False
    if not os.path.exists(file2_path):
        print(f"❌ File not found: {file2_path}")
        return False
    
    hash1 = get_file_hash(file1_path)
    hash2 = get_file_hash(file2_path)
    
    print(f"\nFile 1 hash: {hash1}")
    print(f"File 2 hash: {hash2}")
    
    if hash1 == hash2:
        print("✅ Files are identical")
        return True
    else:
        print("❌ Files are different")
        return False

if __name__ == "__main__":
    # Files to compare (change these if needed)
    file1 = "tinybert_all_training_01.onnx" # 改为第一个ONNX文件的路径（1）
    file2 = "tinybert_all_training_02.onnx" # 改为第二个ONNX文件的路径（2）
    
    print("ONNX File Comparison Tool")
    print("="*40)
    
    # Compare the original files
    print("\n[Comparing original ONNX files]")
    compare_onnx_files(file1, file2)
    
    # Compare their quantized versions
    print("\n[Comparing quantized versions]")
    
    # Create temp quantized files
    quant_file1 = "temp_quant_05.onnx"
    quant_file2 = "temp_quant_06.onnx" 
    
    # Quantize both files with same settings
    quantize_settings = {
        "weight_type": QuantType.QInt8,
        "nodes_to_exclude": ["logits"],
        "extra_options": {
            "WeightSymmetric": False,
            "ActivationSymmetric": False,
            "UseDeterministicCompute": True
        }
    }
    
    print(f"\nQuantizing {file1}...")
    quantize_dynamic(file1, quant_file1, **quantize_settings)
    
    print(f"Quantizing {file2}...")
    quantize_dynamic(file2, quant_file2, **quantize_settings)
    
    # Compare quantized versions
    compare_onnx_files(quant_file1, quant_file2)
    
    # Cleanup
    os.remove(quant_file1)
    os.remove(quant_file2)
    
    print("\nComparison complete. Temporary files removed.")