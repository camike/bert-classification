################################################
# 本文件为Web服务器后端主程序
#
# 功能说明：
# - 提供HTTP服务器和API接口
# - 处理模型训练、转换、量化的后端逻辑
# - 管理文件上传和模型文件操作
# - 实现交互式测试和批量测试功能
################################################

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import cgi
import json
import threading
import time
import shutil
from pathlib import Path
import uuid
import zipfile
import tempfile
import subprocess
import sys
import urllib.parse
import signal
import psutil
import numpy as np

PORT = 8080
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "trained_models"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
PIPELINE_DIR = BASE_DIR / "train_conv_quant"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global state for training progress
training_progress = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'model_id': None,
    'output': '',
    'completed_files': []
}

# Global variable to track training processes
training_processes = {}

# Global variable for batch test progress
batch_test_progress = {'status': 'idle', 'progress': 0, 'message': '', 'results': None}

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/':
                self.serve_template('index.html')
            elif self.path == '/progress':
                self.send_json_response(training_progress)
            elif self.path == '/batch_test_progress':
                self.send_json_response(batch_test_progress)
            elif self.path == '/results':
                self.serve_template('results.html')
            elif self.path.startswith('/download/'):
                self.handle_download()
            elif self.path.startswith('/static/'):
                self.serve_static()
            else:
                self.send_error(404, 'Not Found')
        except Exception as e:
            self.send_error(500, f'Server Error: {str(e)}')

    def do_POST(self):
        try:
            if self.path == '/upload':
                self.handle_upload()
            elif self.path == '/upload_teacher':
                self.handle_upload_teacher()
            elif self.path == '/upload_test_dataset':
                self.handle_test_dataset_upload()
            elif self.path == '/train':
                self.handle_train()
            elif self.path == '/test_model':
                self.handle_test_model()
            elif self.path == '/batch_test':
                self.handle_batch_test()
            elif self.path == '/stop_training':
                self.handle_stop_training()
            else:
                self.send_error(404, 'Not Found')
        except Exception as e:
            self.send_error(500, f'Server Error: {str(e)}')

    def handle_upload(self):
        """Handle file upload"""
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        
        if 'dataset' not in form:
            self.send_error(400, 'No file uploaded')
            return
            
        file_item = form['dataset']
        if not file_item.filename:
            self.send_error(400, 'No file selected')
            return
            
        if not file_item.filename.endswith('.csv'):
            self.send_error(400, 'Only CSV files are supported')
            return
            
        # Save uploaded file as dataset.csv in pipeline directory
        dataset_path = PIPELINE_DIR / "dataset.csv"
        with open(dataset_path, 'wb') as f:
            f.write(file_item.file.read())
        
        self.send_json_response({
            'success': True,
            'message': 'Dataset uploaded successfully'
        })

    def handle_upload_teacher(self):
        """Handle teacher model upload"""
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        
        if 'teacher_model' not in form:
            self.send_json_response({
                'success': False,
                'message': 'No teacher model files uploaded'
            })
            return
        
        try:
            # Create teacher model directory
            teacher_dir = PIPELINE_DIR / "uploaded_teacher_model"
            if teacher_dir.exists():
                shutil.rmtree(teacher_dir)
            os.makedirs(teacher_dir, exist_ok=True)
            
            # Handle multiple files (folder upload) or single file
            teacher_files = form['teacher_model']
            if not isinstance(teacher_files, list):
                teacher_files = [teacher_files]
            
            for file_item in teacher_files:
                if not file_item.filename:
                    continue
                    
                filename = file_item.filename
                
                # Handle compressed files
                if filename.endswith(('.zip', '.tar', '.tar.gz')):
                    # Save compressed file temporarily
                    temp_file = teacher_dir / filename
                    with open(temp_file, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    # Extract compressed file
                    if filename.endswith('.zip'):
                        import zipfile
                        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                            zip_ref.extractall(teacher_dir)
                    elif filename.endswith('.tar') or filename.endswith('.tar.gz'):
                        import tarfile
                        with tarfile.open(temp_file, 'r:*') as tar_ref:
                            tar_ref.extractall(teacher_dir)
                    
                    # Remove temporary compressed file
                    temp_file.unlink()
                else:
                    # Handle individual files (folder upload)
                    file_path = teacher_dir / filename
                    os.makedirs(file_path.parent, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
            
            # Find the actual model directory and update teacher model path
            actual_teacher_path = self.find_teacher_model_path(teacher_dir)
            if actual_teacher_path:
                self.update_teacher_model_config(actual_teacher_path)
                print(f"[DEBUG] Found teacher model at: {actual_teacher_path}")
            else:
                raise Exception("无法找到有效的教师模型文件")
            
            self.send_json_response({
                'success': True,
                'message': 'Teacher model uploaded successfully'
            })
            
        except Exception as e:
            print(f"Teacher model upload failed: {e}")
            self.send_json_response({
                'success': False,
                'message': f'Upload failed: {str(e)}'
            })

    def find_teacher_model_path(self, base_dir):
        """Find the actual teacher model directory containing config.json"""
        # Look for config.json in the uploaded directory structure
        for root, dirs, files in os.walk(base_dir):
            if 'config.json' in files:
                # Found a directory with config.json, return relative path from pipeline dir
                relative_path = Path(root).relative_to(PIPELINE_DIR)
                return str(relative_path)
        
        # If no config.json found, check if there's a single subdirectory
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            # Recursively check the subdirectory
            return self.find_teacher_model_path(subdirs[0])
        
        return None

    def handle_train(self):
        """Handle training request"""
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        
        model_type = post_data.get('model_type', '1')
        use_teacher_model = post_data.get('use_teacher_model', False)
        teacher_model_path = post_data.get('teacher_model_path', '')
        
        # Generate unique model ID
        model_id = uuid.uuid4().hex
        
        print(f"[DEBUG] Training request: model_type={model_type}, use_teacher={use_teacher_model}, teacher_path={teacher_model_path}")
        
        # For distillation model, check teacher model configuration
        if model_type == '3':
            teacher_dir = PIPELINE_DIR / "uploaded_teacher_model"
            has_uploaded_teacher = teacher_dir.exists() and any(teacher_dir.iterdir())
            
            if use_teacher_model and has_uploaded_teacher:
                # Find and update teacher model path
                actual_teacher_path = self.find_teacher_model_path(teacher_dir)
                if actual_teacher_path:
                    self.update_teacher_model_config(actual_teacher_path)
                    print(f"[DEBUG] Using uploaded teacher model: {actual_teacher_path}")
                else:
                    print(f"[DEBUG] Uploaded teacher model invalid, will train BERT first")
            else:
                print(f"[DEBUG] No teacher model specified, will train BERT first")
        
        # Start training in background thread
        thread = threading.Thread(
            target=self.run_training_pipeline,
            args=(model_type, use_teacher_model, teacher_model_path, model_id)
        )
        thread.start()
        
        self.send_json_response({
            'success': True,
            'model_id': model_id
        })

    def run_training_pipeline(self, model_type, use_teacher_model, teacher_model_path, model_id):
        """Run the complete training pipeline using run_all.py"""
        global training_progress, training_processes
        
        original_dir = os.getcwd()
        
        try:
            # Clear any existing files from previous training
            self.cleanup_previous_training_files(model_type)
            
            # Update quantization config BEFORE starting training
            self.update_quantization_config_for_model(model_type)
            
            training_progress.update({
                'status': 'training',
                'progress': 0,
                'message': 'Starting training pipeline...',
                'model_id': model_id,
                'output': '',
                'completed_files': []
            })
            
            # Change to pipeline directory
            os.chdir(str(PIPELINE_DIR))
            
            # Special handling for distilled TinyBERT
            if model_type == '3':
                self.run_distillation_pipeline(model_id, use_teacher_model)
                return
            
            # Prepare input for run_all.py (for BERT and TinyBERT)
            input_text = f"{model_type}\ny\n"
            
            # Add extra confirmations in case the script asks for more input
            input_text += "y\n" * 5  # Add 5 extra 'y' confirmations
            
            print(f"[DEBUG] Starting training with input: {repr(input_text)}")
            
            # Set environment variables to force unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            
            # Run run_all.py with unbuffered output
            process = subprocess.Popen(
                [sys.executable, "-u", "run_all.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=env
            )
            
            # Store process for potential termination
            training_processes[model_id] = process
            
            # Send inputs and close stdin
            process.stdin.write(input_text)
            process.stdin.flush()
            process.stdin.close()
            print(f"[DEBUG] Input sent to process, PID: {process.pid}")
            
            self.monitor_process(process, model_id, model_type)
            
        except Exception as e:
            print(f"[ERROR] Training pipeline failed: {e}")
            training_progress.update({
                'status': 'failed',
                'message': f'Training failed: {str(e)}'
            })
        finally:
            os.chdir(original_dir)
            if model_id in training_processes:
                del training_processes[model_id]

    def run_distillation_pipeline(self, model_id, use_teacher_model=False):
        """Run distillation training pipeline"""
        global training_progress
        
        try:
            # Check if user chose to use teacher model AND teacher model is uploaded
            teacher_dir = PIPELINE_DIR / "uploaded_teacher_model"
            has_uploaded_teacher = teacher_dir.exists() and any(teacher_dir.iterdir())
            
            if use_teacher_model and has_uploaded_teacher:
                # Use uploaded teacher model directly
                training_progress.update({
                    'progress': 10,
                    'message': '使用上传的教师模型，开始蒸馏训练...'
                })
                
                # Skip teacher training, go directly to distillation
                if not self.run_distillation_training():
                    raise Exception("蒸馏训练失败")
                    
            else:
                # No teacher model selected or uploaded, train BERT first
                training_progress.update({
                    'progress': 5,
                    'message': '未选择教师模型，开始训练BERT教师模型...'
                })
                
                if not self.run_bert_training():
                    raise Exception("BERT教师模型训练失败")
                
                training_progress.update({
                    'progress': 40,
                    'message': 'BERT教师模型训练完成，开始蒸馏训练...'
                })
                
                # Update teacher model path to use the newly trained BERT
                self.update_teacher_model_config("trained_model_bert")
                
                # Run distillation training
                if not self.run_distillation_training():
                    raise Exception("蒸馏训练失败")
            
            training_progress.update({
                'progress': 70,
                'message': '蒸馏训练完成，开始ONNX转换...'
            })
            
            # Step 3: Convert to ONNX
            if not self.convert_distilled_to_onnx():
                raise Exception("ONNX转换失败")
            
            training_progress.update({
                'progress': 85,
                'message': 'ONNX转换完成，开始量化...'
            })
            
            # Step 4: Quantize model
            if not self.quantize_distilled_model():
                raise Exception("模型量化失败")
            
            # Step 5: Collect all generated files
            self.collect_generated_files(model_id, '3')
            
            training_progress.update({
                'status': 'completed',
                'progress': 100,
                'message': '蒸馏训练完成！',
                'completed_files': training_progress.get('completed_files', [])
            })
            
        except Exception as e:
            print(f"[ERROR] Distillation pipeline failed: {e}")
            training_progress.update({
                'status': 'failed',
                'message': f'蒸馏训练失败: {str(e)}'
            })

    def run_bert_training(self):
        """Run BERT teacher model training"""
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                [sys.executable, "-u", "train_bert.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            return self.monitor_training_process(process, "BERT教师模型")
            
        except Exception as e:
            print(f"[ERROR] BERT training failed: {e}")
            return False

    def run_distillation_training(self):
        """Run TinyBERT distillation training"""
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                [sys.executable, "-u", "train_tinybert_distilled.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            return self.monitor_training_process(process, "TinyBERT蒸馏")
            
        except Exception as e:
            print(f"[ERROR] Distillation training failed: {e}")
            return False

    def convert_distilled_to_onnx(self):
        """Convert distilled model to ONNX"""
        try:
            # First update conversion.py config for distilled model
            self.update_conversion_config_for_distilled()
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Create a simple conversion script that bypasses the menu
            conversion_script = """import sys
sys.path.append('.')
from conversion import BertClassifier

try:
    classifier = BertClassifier()
    classifier.export_onnx()
    print("ONNX转换完成")
except Exception as e:
    print(f"ONNX转换失败: {e}")
    sys.exit(1)
"""
            
            # Write temporary script
            temp_script = PIPELINE_DIR / "temp_convert_distilled.py"
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(conversion_script)
            
            process = subprocess.Popen(
                [sys.executable, "-u", str(temp_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            result = self.monitor_training_process(process, "ONNX转换")
            
            # Clean up temp script
            if temp_script.exists():
                temp_script.unlink()
                
            return result
            
        except Exception as e:
            print(f"[ERROR] ONNX conversion failed: {e}")
            return False

    def update_conversion_config_for_distilled(self):
        """Update conversion.py config for distilled model"""
        script_path = PIPELINE_DIR / "conversion.py"
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'MODEL_DIR = ' in line and '# 需要改路径（1）' in line:
                    lines[i] = f'MODEL_DIR = "trained_model_tinybert_distilled" # 需要改路径（1）'
                elif 'ONNX_FILENAME = ' in line and '# 需要改路径（3）' in line:
                    lines[i] = f'ONNX_FILENAME = "trained_model_tinybert_distilled.onnx" # 需要改路径（3）'
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            print(f"Failed to update conversion config: {e}")

    def quantize_distilled_model(self):
        """Quantize distilled model"""
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Use the existing dynamic_quantization script
            process = subprocess.Popen(
                [sys.executable, "-u", "dynamic_quantization.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            return self.monitor_training_process(process, "模型量化")
            
        except Exception as e:
            print(f"[ERROR] Model quantization failed: {e}")
            return False

    def monitor_training_process(self, process, stage_name):
        """Monitor a training process with timeout"""
        start_time = time.time()
        last_output_time = time.time()
        timeout = 7200  # 2 hours per stage
        no_output_timeout = 1800  # 30 minutes without output
        
        while True:
            current_time = time.time()
            
            # Check if process finished
            if process.poll() is not None:
                # Read remaining output
                remaining = process.stdout.read()
                if remaining:
                    training_progress['output'] += remaining
                    print(f"[{stage_name}] Final output: {remaining}")
                
                success = process.returncode == 0
                if success:
                    print(f"[DEBUG] {stage_name} completed successfully")
                else:
                    print(f"[ERROR] {stage_name} failed with return code: {process.returncode}")
                return success
            
            # Check timeouts
            if current_time - start_time > timeout:
                print(f"[ERROR] {stage_name} timeout - total time exceeded 2 hours")
                process.terminate()
                return False
                
            if current_time - last_output_time > no_output_timeout:
                print(f"[ERROR] {stage_name} timeout - no output for {no_output_timeout/60:.1f} minutes")
                process.terminate()
                return False
            
            # Read output with better handling
            try:
                import select
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([process.stdout], [], [], 1.0)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            line = line.strip()
                            if line:  # Only process non-empty lines
                                training_progress['output'] += line + '\n'
                                last_output_time = current_time
                                print(f"[{stage_name}] {line}")
                else:
                    # Fallback for Windows
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:  # Only process non-empty lines
                            training_progress['output'] += line + '\n'
                            last_output_time = current_time
                            print(f"[{stage_name}] {line}")
            except Exception as e:
                print(f"[DEBUG] Error reading output: {e}")
                pass
            
            # Print periodic status updates
            elapsed = current_time - start_time
            if int(elapsed) % 5 == 0 and elapsed > 0:  # Every 5 seconds
                print(f"[DEBUG] {stage_name} running for {elapsed/60:.1f} minutes, last output {(current_time - last_output_time)/60:.1f} minutes ago")
            
            time.sleep(0.1)

    def monitor_process(self, process, model_id, model_type):
        """Monitor regular training process"""
        output_lines = []
        last_output_time = time.time()
        no_output_timeout = 600  # 10 minutes without output (increased for distillation)
        total_timeout = 5400  # 90 minutes total (increased for distillation)
        start_time = time.time()
        
        # Read output line by line in real-time
        while True:
            current_time = time.time()
            
            # Check if process is still running
            if process.poll() is not None:
                print(f"[DEBUG] Process finished with return code: {process.returncode}")
                # Process finished, read any remaining output
                remaining_output = process.stdout.read()
                if remaining_output:
                    for line in remaining_output.split('\n'):
                        if line.strip():
                            output_lines.append(line.strip())
                            self.update_progress_from_output(line.strip())
                break
            
            # Check for total timeout
            if current_time - start_time > total_timeout:
                print(f"[DEBUG] Total timeout reached, terminating process")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                raise Exception("Training process timeout - total time exceeded 1 hour")
            
            # Check for no-output timeout
            if current_time - last_output_time > no_output_timeout:
                print(f"[DEBUG] No output timeout reached, terminating process")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                raise Exception("Training process timeout - no output for 5 minutes")
            
            # Try to read a line with timeout
            try:
                # Use select on Unix systems for non-blocking read
                import select
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([process.stdout], [], [], 1.0)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            line = line.strip()
                            if line:
                                print(f"[DEBUG] Got output: {line}")
                                output_lines.append(line)
                                self.update_progress_from_output(line)
                                last_output_time = current_time
                                
                                # Update output display (keep last 200 lines)
                                training_progress.update({
                                    'output': '\n'.join(output_lines[-200:]),
                                    'model_id': model_id
                                })
                        else:
                            # No output available, check if process is still alive
                            if current_time - last_output_time > 30:  # 30 seconds without output
                                print(f"[DEBUG] No output for 30 seconds, process still running: {process.poll() is None}")
                    else:
                        # Fallback for Windows
                        line = process.stdout.readline()
                        if line:
                            line = line.strip()
                            if line:
                                print(f"[DEBUG] Got output: {line}")
                                output_lines.append(line)
                                self.update_progress_from_output(line)
                                last_output_time = current_time
                                
                                # Update output display (keep last 200 lines)
                                training_progress.update({
                                    'output': '\n'.join(output_lines[-200:]),
                                    'model_id': model_id
                                })
                        else:
                            time.sleep(0.1)
                            
            except Exception as e:
                print(f"[DEBUG] Exception in output reading: {e}")
                time.sleep(1)
                continue
        
        # Wait for process to complete
        return_code = process.wait()
        print(f"[DEBUG] Process completed with return code: {return_code}")
        
        if return_code == 0:
            # Copy generated files to model directory
            self.collect_generated_files(model_id, model_type)
            
            training_progress.update({
                'status': 'completed',
                'progress': 100,
                'message': 'Training pipeline completed successfully!',
                'model_id': model_id
            })
        else:
            raise Exception(f"Pipeline failed with return code {return_code}")
            
    def update_teacher_model_config(self, teacher_path):
        """Update teacher model path in distillation script"""
        script_path = PIPELINE_DIR / "train_tinybert_distilled.py"
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'TEACHER_MODEL_PATH = ' in line and '# 修改为教师模型路径（1）' in line:
                    lines[i] = f'    TEACHER_MODEL_PATH = "{teacher_path}" # 修改为教师模型路径（1）'
                    break
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            print(f"[DEBUG] Updated teacher model path to: {teacher_path}")
                
        except Exception as e:
            print(f"Failed to update teacher model config: {e}")

    def collect_generated_files(self, model_id, model_type):
        """Collect all generated files after training completion"""
        completed_files = []
        
        # Update accuracy_summary.py config for testing
        self.update_accuracy_summary_config(model_type)
        
        try:
            if model_type == '1':  # BERT
                model_dir = PIPELINE_DIR / "trained_model_bert"
                onnx_file = PIPELINE_DIR / "trained_model_bert.onnx"
                quantized_file = PIPELINE_DIR / "trained_model_bert_quant.onnx"
                
            elif model_type == '2':  # TinyBERT
                model_dir = PIPELINE_DIR / "trained_model_tinybert"
                onnx_file = PIPELINE_DIR / "trained_model_tinybert.onnx"
                quantized_file = PIPELINE_DIR / "trained_model_tinybert_quant.onnx"
                
            elif model_type == '3':  # TinyBERT Distilled
                model_dir = PIPELINE_DIR / "trained_model_tinybert_distilled"
                onnx_file = PIPELINE_DIR / "trained_model_tinybert_distilled.onnx"
                quantized_file = PIPELINE_DIR / "trained_model_tinybert_distilled_quant.onnx"
            
            files = []
            
            # Add model directory (as a single item)
            if model_dir.exists():
                # Calculate total size of directory
                total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                total_size_mb = total_size / (1024 * 1024)
                files.append({
                    'name': f"{model_dir.name}/",
                    'path': str(model_dir),
                    'size': f"{total_size_mb:.2f}MB",
                    'type': 'pytorch'
                })
            
            # Add ONNX file
            if onnx_file.exists():
                file_size = onnx_file.stat().st_size / (1024 * 1024)
                files.append({
                    'name': onnx_file.name,
                    'path': str(onnx_file),
                    'size': f"{file_size:.2f}MB",
                    'type': 'onnx'
                })
            
            # Add quantized ONNX file
            if quantized_file.exists():
                file_size = quantized_file.stat().st_size / (1024 * 1024)
                files.append({
                    'name': quantized_file.name,
                    'path': str(quantized_file),
                    'size': f"{file_size:.2f}MB",
                    'type': 'quantized'
                })
            
            # Update progress with all files
            training_progress['completed_files'] = files
            
            print(f"[DEBUG] Collected {len(files)} items:")
            for file_info in files:
                print(f"  - {file_info['name']} ({file_info['size']})")
                
        except Exception as e:
            print(f"[ERROR] Failed to collect files: {e}")
            training_progress['completed_files'] = []

    def update_progress_from_output(self, line):
        """Update progress based on output line"""
        global training_progress
        
        # Update output
        current_output = training_progress.get('output', '')
        training_progress['output'] = current_output + line + '\n'
        
        # Parse progress from different stages
        if '开始训练BERT教师模型' in line:
            training_progress.update({
                'progress': 10,
                'message': '正在训练BERT教师模型...'
            })
        elif 'BERT教师模型训练完成' in line:
            training_progress.update({
                'progress': 40,
                'message': 'BERT教师模型训练完成，开始蒸馏训练...'
            })
        elif '开始蒸馏' in line:
            training_progress.update({
                'progress': 45,
                'message': '开始TinyBERT蒸馏训练...'
            })
        elif '蒸馏完成' in line:
            training_progress.update({
                'progress': 70,
                'message': '蒸馏训练完成，开始ONNX转换...'
            })
        elif 'ONNX转换完成' in line:
            training_progress.update({
                'progress': 85,
                'message': 'ONNX转换完成，开始量化...'
            })
        elif '动态量化完成' in line:
            training_progress.update({
                'progress': 95,
                'message': '量化完成，整理文件...'
            })
        elif '完整流水线执行成功' in line:
            training_progress.update({
                'status': 'completed',
                'progress': 100,
                'message': '训练完成！',
                'completed_files': [
                    'trained_model_tinybert_distilled.onnx',
                    'trained_model_tinybert_distilled_quant.onnx'
                ]
            })
        # Handle regular training progress
        elif '训练进度' in line and '%' in line:
            try:
                # Extract percentage from training progress
                import re
                match = re.search(r'(\d+)%', line)
                if match:
                    percent = int(match.group(1))
                    # Scale based on current stage
                    if '教师模型' in training_progress.get('message', ''):
                        scaled_progress = 10 + (percent * 30 // 100)  # 10-40%
                    else:
                        scaled_progress = 45 + (percent * 25 // 100)  # 45-70%
                    training_progress['progress'] = min(scaled_progress, 95)
            except:
                pass

    def handle_download(self):
        """Handle model download"""
        model_id = self.path.split('/')[-1]
        
        # Get the current training progress to find model type
        if not training_progress.get('model_id') or training_progress['model_id'] != model_id:
            self.send_error(404, 'Model not found or training not completed')
            return
        
        # Get completed files from training progress
        completed_files = training_progress.get('completed_files', [])
        if not completed_files:
            self.send_error(404, 'No files available for download')
            return
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    
                    for file_info in completed_files:
                        file_path = Path(file_info['path'])
                        
                        if file_path.exists():
                            if file_path.is_dir():
                                # Add directory contents
                                for sub_file in file_path.rglob('*'):
                                    if sub_file.is_file():
                                        arcname = f"{file_path.name}/{sub_file.relative_to(file_path)}"
                                        zipf.write(sub_file, arcname)
                            else:
                                # Add single file
                                zipf.write(file_path, file_path.name)
                
                with open(temp_zip.name, 'rb') as f:
                    file_data = f.read()
                
                # Determine model type from completed files
                model_name = "trained_model"
                for file_info in completed_files:
                    if file_info['type'] == 'pytorch':
                        model_name = file_info['name'].rstrip('/')
                        break
                
                self.send_response(200)
                self.send_header('Content-type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{model_name}_{model_id}.zip"')
                self.send_header('Content-Length', str(len(file_data)))
                self.end_headers()
                self.wfile.write(file_data)
                
                os.unlink(temp_zip.name)
                print(f"[DEBUG] Download completed for model {model_id}")
                
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            self.send_error(500, f'Download failed: {str(e)}')

    def handle_stop_training(self):
        """Handle stop training request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            model_id = post_data.get('model_id')
            
            if model_id and model_id in training_processes:
                process = training_processes[model_id]
                
                print(f"[DEBUG] Terminating training process for model {model_id}, PID: {process.pid}")
                
                # Try to terminate gracefully first
                try:
                    # Kill the entire process tree
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    
                    # Terminate children first
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    
                    # Terminate parent
                    parent.terminate()
                    
                    # Wait a bit for graceful termination
                    time.sleep(2)
                    
                    # Force kill if still running
                    if parent.is_running():
                        for child in children:
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        parent.kill()
                    
                    # Clean up
                    del training_processes[model_id]
                    
                    # Update training progress
                    global training_progress
                    training_progress.update({
                        'status': 'stopped',
                        'progress': 0,
                        'message': 'Training terminated by user',
                        'model_id': model_id
                    })
                    
                    self.send_json_response({'success': True, 'message': 'Training terminated'})
                    
                except Exception as e:
                    print(f"[DEBUG] Error terminating process: {e}")
                    self.send_json_response({'success': False, 'message': f'Error terminating training: {str(e)}'})
            else:
                self.send_json_response({'success': False, 'message': 'No active training process found'})
                
        except Exception as e:
            print(f"[DEBUG] Error in stop_training: {e}")
            self.send_json_response({'success': False, 'message': str(e)})

    def cleanup_previous_training_files(self, model_type):
        """Clean up previous training files"""
        try:
            if model_type == '1':  # BERT
                cleanup_paths = [
                    PIPELINE_DIR / "trained_model_bert",
                    PIPELINE_DIR / "trained_model_bert.onnx",
                    PIPELINE_DIR / "trained_model_bert_quant.onnx"
                ]
            elif model_type == '2':  # TinyBERT
                cleanup_paths = [
                    PIPELINE_DIR / "trained_model_tinybert",
                    PIPELINE_DIR / "trained_model_tinybert.onnx", 
                    PIPELINE_DIR / "trained_model_tinybert_quant.onnx"
                ]
            elif model_type == '3':  # TinyBERT Distilled
                cleanup_paths = [
                    PIPELINE_DIR / "trained_model_tinybert_distilled",
                    PIPELINE_DIR / "trained_model_tinybert_distilled.onnx",
                    PIPELINE_DIR / "trained_model_tinybert_distilled_quant.onnx"
                ]
            
            for path in cleanup_paths:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"[DEBUG] Cleaned up directory: {path}")
                    else:
                        path.unlink()
                        print(f"[DEBUG] Cleaned up file: {path}")
                        
            print(f"[DEBUG] Cleanup completed for model type {model_type}")
                        
        except Exception as e:
            print(f"[WARNING] Failed to cleanup some files: {e}")

    def serve_template(self, template_name):
        """Serve HTML template"""
        template_path = TEMPLATE_DIR / template_name
        if not template_path.exists():
            self.send_error(404, f'Template {template_name} not found')
            return
            
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f'Error serving template: {str(e)}')

    def serve_static(self):
        """Serve static files"""
        file_path = STATIC_DIR / self.path[8:]  # Remove '/static/' prefix
        if not file_path.exists():
            self.send_error(404, 'Static file not found')
            return
            
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            if file_path.suffix == '.css':
                content_type = 'text/css'
            elif file_path.suffix == '.js':
                content_type = 'application/javascript'
            elif file_path.suffix == '.png':
                content_type = 'image/png'
            elif file_path.suffix == '.jpg' or file_path.suffix == '.jpeg':
                content_type = 'image/jpeg'
            else:
                content_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f'Error serving static file: {str(e)}')

    def send_json_response(self, data):
        """Send JSON response"""
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert numpy types before JSON serialization
        converted_data = convert_numpy_types(data)
        json_data = json.dumps(converted_data, ensure_ascii=False)
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))

    def update_quantization_config_for_model(self, model_type):
        """Update dynamic_quantization.py config for specific model"""
        script_path = PIPELINE_DIR / "dynamic_quantization.py"
        
        # Determine file paths based on model type
        if model_type == '1':  # BERT
            onnx_path = "trained_model_bert.onnx"
            quant_path = "trained_model_bert_quant.onnx"
        elif model_type == '2':  # TinyBERT
            onnx_path = "trained_model_tinybert.onnx"
            quant_path = "trained_model_tinybert_quant.onnx"
        elif model_type == '3':  # TinyBERT Distilled
            onnx_path = "trained_model_tinybert_distilled.onnx"
            quant_path = "trained_model_tinybert_distilled_quant.onnx"
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # Update function default parameter
                if 'def quantize_onnx(model_path, quant_path=' in line:
                    lines[i] = f'def quantize_onnx(model_path, quant_path="{quant_path}"): # 改为量化后的ONNX模型的保存路径（1）'
                # Update input argument default
                elif 'parser.add_argument("--input", default=' in line and '# 改为原始ONNX模型的路径（2）' in line:
                    lines[i] = f'    parser.add_argument("--input", default="{onnx_path}") # 改为原始ONNX模型的路径（2）'
                # Update output argument default
                elif 'parser.add_argument("--output", default=' in line and '# 改为量化后的ONNX模型的保存路径（3）' in line:
                    lines[i] = f'    parser.add_argument("--output", default="{quant_path}") # 改为量化后的ONNX模型的保存路径（3）'
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            print(f"[DEBUG] Updated quantization config:")
            print(f"  Input ONNX: {onnx_path}")
            print(f"  Output quantized: {quant_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update quantization config: {e}")
            return False

    def handle_test_model(self):
        """Handle model testing request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            
            model_id = post_data.get('model_id')
            text = post_data.get('text', '').strip()
            
            if not text:
                self.send_json_response({
                    'success': False,
                    'message': '测试文本不能为空'
                })
                return
            
            # Run model testing
            results = self.run_model_testing(model_id, text)
            
            self.send_json_response({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            print(f"[ERROR] Model testing failed: {e}")
            self.send_json_response({
                'success': False,
                'message': f'测试失败: {str(e)}'
            })

    def run_model_testing(self, model_id, text):
        """Run model testing using accuracy_summary.py logic"""
        try:
            # Change to pipeline directory first
            original_dir = os.getcwd()
            os.chdir(str(PIPELINE_DIR))
            
            # Import the testing logic from accuracy_summary.py
            import sys
            sys.path.insert(0, str(PIPELINE_DIR))
            
            # Import the ONNXModelTester class
            import importlib.util
            spec = importlib.util.spec_from_file_location("accuracy_summary", PIPELINE_DIR / "accuracy_summary.py")
            accuracy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(accuracy_module)
            
            # Create tester instance (this will automatically load the correct model paths from the config)
            tester = accuracy_module.ONNXModelTester()
            
            results = {}
            
            # Test both models using the predict method from accuracy_summary.py
            for model_type in ['original', 'quantized']:
                if model_type in tester.sessions:
                    try:
                        pred, probs = tester.predict(text, model_type)
                        
                        # Format probabilities using the label_map from accuracy_summary.py
                        prob_dict = {}
                        for i, prob in enumerate(probs):
                            label = tester.label_map.get(i, f"Label_{i}")
                            prob_dict[label] = float(prob)
                        
                        results[model_type] = {
                            'prediction': int(pred),
                            'label': tester.label_map.get(pred, 'Unknown'),
                            'probabilities': prob_dict
                        }
                    except Exception as e:
                        print(f"[WARNING] Failed to test {model_type} model: {e}")
                        continue
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Model testing execution failed: {e}")
            raise e
        finally:
            # Always restore original directory
            os.chdir(original_dir)
            # Clean up sys.path
            if str(PIPELINE_DIR) in sys.path:
                sys.path.remove(str(PIPELINE_DIR))

    def update_accuracy_summary_config(self, model_type):
        """Update accuracy_summary.py config for current model"""
        script_path = PIPELINE_DIR / "accuracy_summary.py"
        
        # Determine file paths based on model type (use relative paths)
        if model_type == '1':  # BERT
            model_dir = "trained_model_bert"
            onnx_path = "trained_model_bert.onnx"
            quant_path = "trained_model_bert_quant.onnx"
        elif model_type == '2':  # TinyBERT
            model_dir = "trained_model_tinybert"
            onnx_path = "trained_model_tinybert.onnx"
            quant_path = "trained_model_tinybert_quant.onnx"
        elif model_type == '3':  # TinyBERT Distilled
            model_dir = "trained_model_tinybert_distilled"
            onnx_path = "trained_model_tinybert_distilled.onnx"
            quant_path = "trained_model_tinybert_distilled_quant.onnx"
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'MODEL_DIR = ' in line and '# 改为原始模型文件夹路径（1）' in line:
                    lines[i] = f'MODEL_DIR = "{model_dir}" # 改为原始模型文件夹路径（1）'
                elif 'ONNX_MODEL_PATH = ' in line and '# 改为量化前ONNX文件路径（2）' in line:
                    lines[i] = f'ONNX_MODEL_PATH = "{onnx_path}" # 改为量化前ONNX文件路径（2）'
                elif 'QUANT_MODEL_PATH = ' in line and '# 改为量化后ONNX文件路径（3）' in line:
                    lines[i] = f'QUANT_MODEL_PATH = "{quant_path}" # 改为量化后ONNX文件路径（3）'
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            print(f"[DEBUG] Updated accuracy_summary config for model type {model_type}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update accuracy_summary config: {e}")
            return False

    def handle_test_dataset_upload(self):
        """Handle test dataset upload"""
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        
        if 'test_dataset' not in form:
            self.send_error(400, 'No file uploaded')
            return
            
        file_item = form['test_dataset']
        if not file_item.filename:
            self.send_error(400, 'No file selected')
            return
            
        if not file_item.filename.endswith('.csv'):
            self.send_error(400, 'Only CSV files are supported')
            return
            
        # Save uploaded file as test_dataset.csv in pipeline directory
        test_dataset_path = PIPELINE_DIR / "test_dataset.csv"
        with open(test_dataset_path, 'wb') as f:
            f.write(file_item.file.read())
        
        self.send_json_response({
            'success': True,
            'message': 'Test dataset uploaded successfully'
        })

    def handle_batch_test(self):
        """Handle batch test request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            
            model_id = post_data.get('model_id')
            
            # Start batch testing in background
            import threading
            test_thread = threading.Thread(target=self.run_batch_test, args=(model_id,))
            test_thread.daemon = True
            test_thread.start()
            
            self.send_json_response({
                'success': True,
                'message': 'Batch test started'
            })
            
        except Exception as e:
            print(f"[ERROR] Batch test failed: {e}")
            self.send_json_response({
                'success': False,
                'message': f'批量测试启动失败: {str(e)}'
            })

    def run_batch_test(self, model_id):
        """Run batch testing using batch_testing.py"""
        global batch_test_progress
        
        def update_progress(progress, message):
            """Progress callback function"""
            batch_test_progress.update({
                'status': 'running',
                'progress': progress,
                'message': message
            })
        
        try:
            # Change to pipeline directory first
            original_dir = os.getcwd()
            os.chdir(str(PIPELINE_DIR))
            
            batch_test_progress = {
                'status': 'running',
                'progress': 10,
                'message': '初始化测试环境...',
                'results': None
            }
            
            # Import the testing logic from batch_testing.py
            import sys
            sys.path.insert(0, str(PIPELINE_DIR))
            
            # Import the DatasetTester class
            import importlib.util
            spec = importlib.util.spec_from_file_location("batch_testing", PIPELINE_DIR / "batch_testing.py")
            testing_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(testing_module)
            
            update_progress(30, '加载模型...')
            
            # Create tester instance
            tester = testing_module.DatasetTester()
            
            update_progress(50, '开始批量测试...')
            
            # Run evaluation with progress callback
            results = tester.evaluate_dataset("test_dataset.csv", return_errors=True, progress_callback=update_progress)
            
            batch_test_progress = {
                'status': 'completed',
                'progress': 100,
                'message': '测试完成!',
                'results': results
            }
            
            print(f"[DEBUG] Batch test completed, results: {len(results) if results else 0} models")
            
        except Exception as e:
            print(f"[ERROR] Batch test execution failed: {e}")
            batch_test_progress = {
                'status': 'error',
                'progress': 0,
                'message': f'测试失败: {str(e)}',
                'results': None
            }
        finally:
            # Restore original directory
            os.chdir(original_dir)
            # Clean up sys.path
            if str(PIPELINE_DIR) in sys.path:
                sys.path.remove(str(PIPELINE_DIR))

def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Server running at http://localhost:{PORT}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
