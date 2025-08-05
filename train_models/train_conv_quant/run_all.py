################################################
# 一键训练+转换+量化工具
#
# 功能说明：
# - 用户选择模型类型（BERT/TinyBERT/TinyBERT蒸馏）
# - 自动完成：训练 → ONNX转换 → 动态量化
# - 统一的文件命名规则
################################################

import os
import sys
import subprocess
import shutil
import torch
import numpy as np
from datetime import datetime
from colorama import Fore, Style, init

# 初始化colorama
init(autoreset=True)

class ModelTrainingPipeline:
    def __init__(self):
        self.models_config = {
            "bert": {
                "name": "BERT",
                "script": "train_bert.py",
                "model_dir": "trained_model_bert",
                "output_prefix": "trained_model_bert"
            },
            "tinybert": {
                "name": "TinyBERT",
                "script": "train_tinybert.py", 
                "model_dir": "trained_model_tinybert",
                "output_prefix": "trained_model_tinybert"
            },
            "tinybert_distilled": {
                "name": "TinyBERT蒸馏",
                "script": "train_tinybert_distilled.py",
                "model_dir": "trained_model_tinybert_distilled",
                "output_prefix": "trained_model_tinybert_distilled"
            }
        }
        
        # 标签映射
        self.label_mapping = {
            0: "其他",
            1: "爱奇艺", 
            2: "飞书",
            3: "鲁大师"
        }
    
    def print_banner(self):
        """打印欢迎横幅"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}🚀 一键训练+转换+量化工具")
        print(f"{Fore.CYAN}📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.CYAN}{'='*60}")
    
    def show_menu(self):
        """显示模型选择菜单"""
        print(f"\n{Fore.YELLOW}请选择要训练的模型:")
        print(f"{Fore.GREEN}1. BERT (bert-base-chinese)")
        print(f"{Fore.GREEN}2. TinyBERT (huawei-noah/TinyBERT_4L_zh)")
        print(f"{Fore.GREEN}3. TinyBERT蒸馏 (需要先训练BERT作为教师模型)")
        print(f"{Fore.RED}4. 退出")
        
        while True:
            choice = input(f"\n{Fore.WHITE}请输入选择 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return choice
            print(f"{Fore.RED}无效输入，请重新选择!")
    
    def run_training(self, model_key):
        """运行训练脚本"""
        config = self.models_config[model_key]
        script_path = config["script"]
        
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}🏋️ 开始训练 {config['name']} 模型")
        print(f"{Fore.CYAN}📄 执行脚本: {script_path}")
        print(f"{Fore.CYAN}{'='*50}")
        
        if not os.path.exists(script_path):
            print(f"{Fore.RED}❌ 训练脚本不存在: {script_path}")
            return False
        
        try:
            # 运行训练脚本
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=False, 
                                  text=True)
            
            if result.returncode == 0:
                print(f"\n{Fore.GREEN}✅ {config['name']} 训练完成!")
                
                # 特殊处理蒸馏模型：复制tokenizer文件到主目录
                if model_key == 'tinybert_distilled':
                    self.fix_distilled_model_files(config['model_dir'])
                
                return True
            else:
                print(f"\n{Fore.RED}❌ {config['name']} 训练失败!")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ 训练过程中出错: {e}")
            return False
    
    def fix_distilled_model_files(self, model_dir):
        """修复蒸馏模型文件结构，确保conversion.py能正常工作"""
        print(f"{Fore.YELLOW}🔧 修复蒸馏模型文件结构...")
        
        try:
            final_model_dir = os.path.join(model_dir, "final_model")
            
            if os.path.exists(final_model_dir):
                # 需要复制的文件列表
                files_to_copy = [
                    "vocab.txt",
                    "tokenizer.json", 
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin"
                ]
                
                copied_files = []
                for filename in files_to_copy:
                    src_path = os.path.join(final_model_dir, filename)
                    dst_path = os.path.join(model_dir, filename)
                    
                    if os.path.exists(src_path) and not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(filename)
                
                if copied_files:
                    print(f"{Fore.GREEN}✅ 已复制文件到主目录: {', '.join(copied_files)}")
                else:
                    print(f"{Fore.YELLOW}⚠️ 未找到需要复制的文件或文件已存在")
            else:
                print(f"{Fore.YELLOW}⚠️ 未找到final_model目录: {final_model_dir}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ 修复文件结构失败: {e}")
    
    def update_conversion_config(self, model_key):
        """更新conversion.py的配置"""
        config = self.models_config[model_key]
        model_dir = config["model_dir"]
        output_prefix = config["output_prefix"]
        onnx_filename = f"{output_prefix}.onnx"
        
        print(f"{Fore.YELLOW}🔧 更新conversion.py配置...")
        
        try:
            # 读取conversion.py文件
            with open("conversion.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            # 更新配置项
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 更新模型目录
                if 'MODEL_DIR = ' in line and '# 需要改路径（1）' in line:
                    lines[i] = f'MODEL_DIR = "{model_dir}" # 需要改路径（1）'
                # 更新ONNX输出目录
                elif 'ONNX_OUTPUT_DIR = ' in line and '# 需要改路径（2）' in line:
                    lines[i] = f'ONNX_OUTPUT_DIR = "." # 需要改路径（2）'
                # 更新ONNX文件名
                elif 'ONNX_FILENAME = ' in line and '# 需要改路径（3）' in line:
                    lines[i] = f'ONNX_FILENAME = "{onnx_filename}" # 需要改路径（3）'
            
            # 写回文件
            with open("conversion.py", "w", encoding="utf-8") as f:
                f.write('\n'.join(lines))
            
            print(f"{Fore.GREEN}✅ conversion.py配置已更新")
            print(f"{Fore.CYAN}  模型目录: {model_dir}")
            print(f"{Fore.CYAN}  ONNX文件: {onnx_filename}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 更新conversion.py配置失败: {e}")
            return False
    
    def update_quantization_config(self, model_key):
        """更新dynamic_quantization.py的配置"""
        config = self.models_config[model_key]
        output_prefix = config["output_prefix"]
        onnx_path = f"{output_prefix}.onnx"
        quant_path = f"{output_prefix}_quant.onnx"
        
        print(f"{Fore.YELLOW}🔧 更新dynamic_quantization.py配置...")
        
        try:
            # 读取dynamic_quantization.py文件
            with open("dynamic_quantization.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            # 更新配置项
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 更新量化模型保存路径（函数参数默认值）
                if 'def quantize_onnx(model_path, quant_path=' in line:
                    lines[i] = f'def quantize_onnx(model_path, quant_path="{quant_path}"): # 改为量化后的ONNX模型的保存路径（1）'
                # 更新原始ONNX模型路径
                elif 'parser.add_argument("--input", default=' in line and '# 改为原始ONNX模型的路径（2）' in line:
                    lines[i] = f'    parser.add_argument("--input", default="{onnx_path}") # 改为原始ONNX模型的路径（2）'
                # 更新量化后ONNX模型路径
                elif 'parser.add_argument("--output", default=' in line and '# 改为量化后的ONNX模型的保存路径（3）' in line:
                    lines[i] = f'    parser.add_argument("--output", default="{quant_path}") # 改为量化后的ONNX模型的保存路径（3）'
            
            # 写回文件
            with open("dynamic_quantization.py", "w", encoding="utf-8") as f:
                f.write('\n'.join(lines))
            
            print(f"{Fore.GREEN}✅ dynamic_quantization.py配置已更新")
            print(f"{Fore.CYAN}  输入ONNX: {onnx_path}")
            print(f"{Fore.CYAN}  输出量化: {quant_path}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 更新dynamic_quantization.py配置失败: {e}")
            return False
    
    def update_quantization_config(self, model_key):
        """更新dynamic_quantization.py的配置"""
        config = self.models_config[model_key]
        output_prefix = config["output_prefix"]
        onnx_path = f"{output_prefix}.onnx"
        quant_path = f"{output_prefix}_quant.onnx"
        
        print(f"{Fore.YELLOW}� 更新dynamic_quantization.py配置...")
        
        try:
            # 读取dynamic_quantization.py文件
            with open("dynamic_quantization.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            # 更新配置项
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 更新量化模型保存路径（函数参数默认值）
                if 'def quantize_onnx(model_path, quant_path=' in line:
                    lines[i] = f'def quantize_onnx(model_path, quant_path="{quant_path}"): # 改为量化后的ONNX模型的保存路径（1）'
                # 更新原始ONNX模型路径
                elif 'parser.add_argument("--input", default=' in line and '# 改为原始ONNX模型的路径（2）' in line:
                    lines[i] = f'    parser.add_argument("--input", default="{onnx_path}") # 改为原始ONNX模型的路径（2）'
                # 更新量化后ONNX模型路径
                elif 'parser.add_argument("--output", default=' in line and '# 改为量化后的ONNX模型的保存路径（3）' in line:
                    lines[i] = f'    parser.add_argument("--output", default="{quant_path}") # 改为量化后的ONNX模型的保存路径（3）'
            
            # 写回文件
            with open("dynamic_quantization.py", "w", encoding="utf-8") as f:
                f.write('\n'.join(lines))
            
            print(f"{Fore.GREEN}✅ dynamic_quantization.py配置已更新")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 更新dynamic_quantization.py配置失败: {e}")
            return False
    
    def convert_to_onnx(self, model_key):
        """使用conversion.py转换为ONNX格式"""
        config = self.models_config[model_key]
        
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}🔄 开始ONNX转换")
        print(f"{Fore.CYAN}📄 使用脚本: conversion.py")
        print(f"{Fore.CYAN}{'='*50}")
        
        # 检查模型目录是否存在
        if not os.path.exists(config["model_dir"]):
            print(f"{Fore.RED}❌ 模型目录不存在: {config['model_dir']}")
            return False
        
        # 更新conversion.py配置
        if not self.update_conversion_config(model_key):
            return False
        
        try:
            # 运行conversion.py with input to skip menu
            print(f"{Fore.YELLOW}🔧 执行ONNX转换...")
            
            # Create a simple script that just does the conversion without menu
            conversion_script = f"""
import sys
sys.path.append('.')
from conversion import BertClassifier

# Just do the conversion and exit
classifier = BertClassifier()
print("ONNX转换完成，自动退出...")
"""
            
            # Write temporary script
            with open("temp_conversion.py", "w", encoding="utf-8") as f:
                f.write(conversion_script)
            
            result = subprocess.run([sys.executable, "temp_conversion.py"], 
                                  capture_output=False, 
                                  text=True)
            
            # Clean up temp file
            if os.path.exists("temp_conversion.py"):
                os.remove("temp_conversion.py")
            
            if result.returncode == 0:
                onnx_path = f"{config['output_prefix']}.onnx"
                if os.path.exists(onnx_path):
                    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                    print(f"\n{Fore.GREEN}✅ ONNX转换完成: {onnx_path}")
                    print(f"{Fore.GREEN}📊 文件大小: {file_size:.2f} MB")
                    return True
                else:
                    print(f"{Fore.RED}❌ ONNX文件未生成: {onnx_path}")
                    return False
            else:
                print(f"\n{Fore.RED}❌ ONNX转换失败!")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ ONNX转换过程中出错: {e}")
            return False
    
    def quantize_model(self, model_key):
        """使用dynamic_quantization.py进行动态量化"""
        config = self.models_config[model_key]
        onnx_path = f"{config['output_prefix']}.onnx"
        quant_path = f"{config['output_prefix']}_quant.onnx"
        
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}⚡ 开始动态量化")
        print(f"{Fore.CYAN}📄 使用脚本: dynamic_quantization.py")
        print(f"{Fore.CYAN}📄 输入文件: {onnx_path}")
        print(f"{Fore.CYAN}📄 输出文件: {quant_path}")
        print(f"{Fore.CYAN}{'='*50}")
        
        if not os.path.exists(onnx_path):
            print(f"{Fore.RED}❌ ONNX文件不存在: {onnx_path}")
            return False
        
        # 更新dynamic_quantization.py配置
        if not self.update_quantization_config(model_key):
            return False
        
        try:
            # 运行dynamic_quantization.py
            print(f"{Fore.YELLOW}🔧 执行INT8量化...")
            result = subprocess.run([sys.executable, "dynamic_quantization.py"], 
                                  capture_output=False, 
                                  text=True)
            
            if result.returncode == 0:
                if os.path.exists(quant_path):
                    # 比较文件大小
                    original_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                    quantized_size = os.path.getsize(quant_path) / (1024 * 1024)  # MB
                    compression_ratio = original_size / quantized_size
                    
                    print(f"\n{Fore.GREEN}✅ 动态量化完成: {quant_path}")
                    print(f"{Fore.GREEN}📊 原始模型: {original_size:.2f} MB")
                    print(f"{Fore.GREEN}📊 量化模型: {quantized_size:.2f} MB")
                    print(f"{Fore.GREEN}📊 压缩比例: {compression_ratio:.2f}x")
                    return True
                else:
                    print(f"{Fore.RED}❌ 量化文件未生成: {quant_path}")
                    return False
            else:
                print(f"\n{Fore.RED}❌ 动态量化失败!")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}❌ 动态量化过程中出错: {e}")
            return False
    
    def show_generated_files(self, model_key):
        """显示生成的文件"""
        config = self.models_config[model_key]
        output_prefix = config["output_prefix"]
        
        print(f"\n{Fore.GREEN}📁 生成的文件:")
        
        files_to_check = [
            (f"{output_prefix}.onnx", "ONNX模型"),
            (f"{output_prefix}_quant.onnx", "量化ONNX模型"),
            (config["model_dir"], "训练模型目录")
        ]
        
        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"{Fore.GREEN}  ✅ {description}: {file_path} ({size:.2f} MB)")
                else:
                    print(f"{Fore.GREEN}  ✅ {description}: {file_path}")
            else:
                print(f"{Fore.RED}  ❌ {description}: {file_path} (未找到)")

    def run_pipeline(self, model_key):
        """运行完整流水线"""
        config = self.models_config[model_key]
        
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.MAGENTA}🎯 开始 {config['name']} 完整流水线")
        print(f"{Fore.MAGENTA}📋 流程: 训练 → ONNX转换 → 动态量化")
        print(f"{Fore.MAGENTA}{'='*60}")
        
        start_time = datetime.now()
        
        # 步骤1: 训练
        print(f"\n{Fore.BLUE}>>> 步骤 1/3: 模型训练 <<<")
        if not self.run_training(model_key):
            print(f"{Fore.RED}❌ 流水线中断: 训练失败")
            return False
        
        # 步骤2: ONNX转换
        print(f"\n{Fore.BLUE}>>> 步骤 2/3: ONNX转换 <<<")
        if not self.convert_to_onnx(model_key):
            print(f"{Fore.RED}❌ 流水线中断: ONNX转换失败")
            return False
        
        # 步骤3: 动态量化
        print(f"\n{Fore.BLUE}>>> 步骤 3/3: 动态量化 <<<")
        if not self.quantize_model(model_key):
            print(f"{Fore.RED}❌ 流水线中断: 动态量化失败")
            return False
        
        # 完成
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}🎉 {config['name']} 完整流水线执行成功!")
        print(f"{Fore.GREEN}⏱️ 总耗时: {duration}")
        print(f"{Fore.GREEN}{'='*60}")
        
        # 显示生成的文件
        self.show_generated_files(model_key)
        
        return True
    
    def run(self):
        """运行主程序"""
        self.print_banner()
        
        # Check if running in automated mode
        import sys
        if not sys.stdin.isatty():
            print(f"{Fore.YELLOW}检测到自动化模式，等待输入...")
            try:
                # Read the first line for model choice
                choice = input().strip()
                if choice in ['1', '2', '3']:
                    model_mapping = {
                        '1': 'bert',
                        '2': 'tinybert', 
                        '3': 'tinybert_distilled'
                    }
                    model_key = model_mapping[choice]
                    
                    # Read confirmation
                    confirm = input().strip().lower()
                    if confirm == 'y':
                        print(f"{Fore.GREEN}开始自动化流水线...")
                        self.run_pipeline(model_key)
                        return
                    else:
                        print(f"{Fore.YELLOW}用户取消执行")
                        return
                else:
                    print(f"{Fore.RED}无效的模型选择: {choice}")
                    return
            except EOFError:
                print(f"{Fore.RED}自动化模式下输入不足")
                return
        
        # Interactive mode
        while True:
            choice = self.show_menu()
            
            if choice == '4':
                print(f"\n{Fore.CYAN}👋 感谢使用，再见!")
                break
            
            # 映射选择到模型键
            model_mapping = {
                '1': 'bert',
                '2': 'tinybert', 
                '3': 'tinybert_distilled'
            }
            
            model_key = model_mapping[choice]
            
            # 特殊处理蒸馏模型
            if model_key == 'tinybert_distilled':
                print(f"\n{Fore.YELLOW}⚠️ 蒸馏训练需要教师模型!")
                
                # 检查并处理教师模型
                teacher_path = self.check_teacher_model()
                if not teacher_path:
                    print(f"{Fore.YELLOW}❌ 未配置教师模型，已取消蒸馏训练")
                    continue
                
                # 更新蒸馏训练配置
                if not self.update_distillation_config(teacher_path):
                    print(f"{Fore.RED}❌ 配置更新失败，已取消蒸馏训练")
                    continue
            
            # 确认执行
            config = self.models_config[model_key]
            print(f"\n{Fore.YELLOW}📋 即将执行 {config['name']} 完整流水线")
            print(f"{Fore.YELLOW}📄 预计生成文件:")
            print(f"{Fore.YELLOW}  - {config['output_prefix']}.onnx")
            print(f"{Fore.YELLOW}  - {config['output_prefix']}_quant.onnx")
            
            confirm = input(f"\n{Fore.WHITE}确认执行? (y/n): ").lower()
            if confirm == 'y':
                self.run_pipeline(model_key)
            else:
                print(f"{Fore.YELLOW}❌ 已取消执行")
            
            # 询问是否继续
            continue_choice = input(f"\n{Fore.WHITE}是否继续训练其他模型? (y/n): ").lower()
            if continue_choice != 'y':
                print(f"\n{Fore.CYAN}👋 感谢使用，再见!")
                break

    def check_teacher_model(self):
        """检查教师模型"""
        print(f"\n{Fore.CYAN}🔍 检查教师模型...")
        
        # 不再检查默认路径，直接询问用户
        print(f"{Fore.YELLOW}蒸馏训练需要BERT教师模型")
        print(f"{Fore.YELLOW}选项:")
        print(f"{Fore.GREEN}  1. 先训练BERT教师模型（推荐）")
        print(f"{Fore.RED}  2. 取消蒸馏训练")
        
        choice = input(f"\n{Fore.WHITE}请选择 (1-2): ").strip()
        if choice == '1':
            return self.train_teacher_model()
        else:
            return None

    def train_teacher_model(self):
        """训练BERT教师模型"""
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.CYAN}👨‍🏫 开始训练BERT教师模型")
        print(f"{Fore.CYAN}{'='*50}")
        
        # 运行BERT训练
        if self.run_training('bert'):
            teacher_path = self.models_config['bert']['model_dir']
            print(f"{Fore.GREEN}✅ BERT教师模型训练完成: {teacher_path}")
            return teacher_path
        else:
            print(f"{Fore.RED}❌ BERT教师模型训练失败")
            return None

    def update_distillation_config(self, teacher_path):
        """更新蒸馏训练脚本的教师模型路径"""
        print(f"{Fore.YELLOW}🔧 更新蒸馏训练配置...")
        
        try:
            with open("train_tinybert_distilled.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'TEACHER_MODEL_PATH = ' in line and '# 修改为教师模型路径（1）' in line:
                    lines[i] = f'    TEACHER_MODEL_PATH = "{teacher_path}" # 修改为教师模型路径（1）'
                    break
            
            with open("train_tinybert_distilled.py", "w", encoding="utf-8") as f:
                f.write('\n'.join(lines))
            
            print(f"{Fore.GREEN}✅ 蒸馏训练配置已更新")
            print(f"{Fore.CYAN}  教师模型路径: {teacher_path}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ 更新蒸馏训练配置失败: {e}")
            return False

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run()
