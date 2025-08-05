////////////////////////////////////////////////
// 本文件为Web界面的前端JavaScript脚本
//
// 功能说明：
// - 处理用户界面交互和事件响应
// - 管理模型训练、转换、量化的前端流程
// - 实时显示训练进度和批量测试结果
// - 提供交互式测试功能的前端逻辑
////////////////////////////////////////////////

let currentModelId = null;
let progressInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    const datasetFile = document.getElementById('datasetFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const backHomeBtn = document.getElementById('backHomeBtn');
    const useTeacherModel = document.getElementById('useTeacherModel');
    const modelTypeRadios = document.querySelectorAll('input[name="model_type"]');
    const teacherModelFile = document.getElementById('teacherModelFile');
    const uploadTeacherBtn = document.getElementById('uploadTeacherBtn');

    // File selection handling
    datasetFile.addEventListener('change', function() {
        const file = this.files[0];
        const fileInfo = document.getElementById('fileInfo');
        
        if (file) {
            if (!file.name.endsWith('.csv')) {
                showStatus('uploadStatus', '仅支持CSV格式文件', 'error');
                this.value = '';
                uploadBtn.disabled = true;
                fileInfo.textContent = '未选择文件';
                return;
            }
            
            const fileSizeMB = file.size / (1024 * 1024);
            if (fileSizeMB > 100) {
                showStatus('uploadStatus', '文件大小不能超过100MB', 'error');
                this.value = '';
                uploadBtn.disabled = true;
                fileInfo.textContent = '未选择文件';
                return;
            }
            
            fileInfo.textContent = `已选择: ${file.name} (${fileSizeMB.toFixed(2)}MB)`;
            uploadBtn.disabled = false;
            showStatus('uploadStatus', '', '');
        } else {
            fileInfo.textContent = '未选择文件';
            uploadBtn.disabled = true;
        }
    });

    // Model type change handling
    modelTypeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            const teacherConfig = document.getElementById('teacherModelConfig');
            if (this.value === '3') {
                teacherConfig.style.display = 'block';
            } else {
                teacherConfig.style.display = 'none';
            }
            updateStartTrainingButtonState();
        });
    });

    // Teacher model file selection handling
    teacherModelFile.addEventListener('change', function() {
        const files = this.files;
        const fileInfo = document.getElementById('teacherFileInfo');
        
        if (files.length > 0) {
            if (files.length === 1 && (files[0].name.endsWith('.zip') || files[0].name.endsWith('.tar') || files[0].name.endsWith('.tar.gz'))) {
                // Single compressed file
                const file = files[0];
                const fileSizeMB = file.size / (1024 * 1024);
                fileInfo.textContent = `已选择: ${file.name} (${fileSizeMB.toFixed(2)}MB)`;
                uploadTeacherBtn.disabled = false;
            } else if (files.length > 1) {
                // Multiple files (folder)
                let totalSize = 0;
                for (let file of files) {
                    totalSize += file.size;
                }
                const totalSizeMB = totalSize / (1024 * 1024);
                fileInfo.textContent = `已选择: ${files.length}个文件 (${totalSizeMB.toFixed(2)}MB)`;
                uploadTeacherBtn.disabled = false;
            } else {
                fileInfo.textContent = '请选择教师模型文件夹或压缩包';
                uploadTeacherBtn.disabled = true;
            }
            showStatus('teacherUploadStatus', '', '');
        } else {
            fileInfo.textContent = '未选择文件';
            uploadTeacherBtn.disabled = true;
        }
    });

    // Teacher model upload button click
    uploadTeacherBtn.addEventListener('click', function() {
        const files = teacherModelFile.files;
        if (files.length === 0) {
            showStatus('teacherUploadStatus', '请选择教师模型文件', 'error');
            return;
        }

        const formData = new FormData();
        
        // Add all files to form data
        for (let i = 0; i < files.length; i++) {
            formData.append('teacher_model', files[i]);
        }

        showStatus('teacherUploadStatus', '上传中...', 'loading');
        uploadTeacherBtn.disabled = true;

        fetch('/upload_teacher', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showStatus('teacherUploadStatus', '✅ 教师模型上传成功!', 'success');
                // Re-enable start training button after successful upload
                updateStartTrainingButtonState();
            } else {
                showStatus('teacherUploadStatus', '❌ 上传失败: ' + (data.message || ''), 'error');
                uploadTeacherBtn.disabled = false;
            }
        })
        .catch(error => {
            showStatus('teacherUploadStatus', `❌ 上传失败: ${error.message}`, 'error');
            uploadTeacherBtn.disabled = false;
        });
    });

    // Teacher model checkbox handling
    useTeacherModel.addEventListener('change', function() {
        const teacherPath = document.getElementById('teacherModelPath');
        if (this.checked) {
            teacherPath.style.display = 'block';
        } else {
            teacherPath.style.display = 'none';
        }
        updateStartTrainingButtonState();
    });

    // Upload button click
    uploadBtn.addEventListener('click', function() {
        const file = datasetFile.files[0];
        if (!file) {
            showStatus('uploadStatus', '请选择文件', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('dataset', file);

        showStatus('uploadStatus', '上传中...', 'loading');
        uploadBtn.disabled = true;

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showStatus('uploadStatus', '✅ 数据集上传成功!', 'success');
                document.getElementById('modelSelectionCard').style.display = 'block';
                document.getElementById('modelSelectionCard').scrollIntoView({ behavior: 'smooth' });
            } else {
                showStatus('uploadStatus', '❌ 上传失败', 'error');
                uploadBtn.disabled = false;
            }
        })
        .catch(error => {
            showStatus('uploadStatus', `❌ 上传失败: ${error.message}`, 'error');
            uploadBtn.disabled = false;
        });
    });

    // Start training button click
    startTrainingBtn.addEventListener('click', function() {
        const selectedModel = document.querySelector('input[name="model_type"]:checked');
        if (!selectedModel) {
            alert('请选择模型类型');
            return;
        }
        
        const useTeacher = document.getElementById('useTeacherModel').checked;

        // For distillation model (type 3), check if teacher model is required and uploaded
        if (selectedModel.value === '3' && useTeacher) {
            const teacherUploadStatus = document.getElementById('teacherUploadStatus');
            if (!teacherUploadStatus.textContent.includes('✅')) {
                alert('请先上传教师模型');
                return;
            }
        }

        const requestData = {
            model_type: selectedModel.value,
            use_teacher_model: useTeacher,
            teacher_model_path: 'uploaded_teacher_model'  // Fixed path for uploaded models
        };

        // 显示进度卡片
        document.getElementById('progressCard').style.display = 'block';
        document.getElementById('progressCard').scrollIntoView({ behavior: 'smooth' });
        
        showStatus('statusMessage', '启动训练...', 'loading');
        startTrainingBtn.disabled = true;

        fetch('/train', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentModelId = data.model_id;
                startProgressMonitoring();
            } else {
                showError(data.message || '启动训练失败');
                startTrainingBtn.disabled = false;
            }
        })
        .catch(error => {
            showError('网络错误: ' + error.message);
            startTrainingBtn.disabled = false;
        });
    });

    // Download button click
    downloadBtn.addEventListener('click', function() {
        if (currentModelId) {
            // Create a zip download of all files
            window.location.href = `/download/${currentModelId}`;
        }
    });

    // Back home button click
    backHomeBtn.addEventListener('click', function() {
        location.reload();
    });

    // Function to update start training button state
    function updateStartTrainingButtonState() {
        const selectedModel = document.querySelector('input[name="model_type"]:checked');
        const useTeacher = document.getElementById('useTeacherModel').checked;
        const teacherUploadStatus = document.getElementById('teacherUploadStatus');
        
        if (selectedModel && selectedModel.value === '3' && useTeacher) {
            // For distillation with teacher model, check if upload is successful
            startTrainingBtn.disabled = !teacherUploadStatus.textContent.includes('✅');
        } else {
            // For other cases, enable if model is selected
            startTrainingBtn.disabled = !selectedModel;
        }
    }

    // Testing module event listeners
    const interactiveTestBtn = document.getElementById('interactiveTestBtn');
    const testingPanel = document.getElementById('testingPanel');
    const testChatInput = document.getElementById('testChatInput');
    const sendTestBtn = document.getElementById('sendTestBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const finishTestBtn = document.getElementById('finishTestBtn');
    
    if (interactiveTestBtn) {
        interactiveTestBtn.addEventListener('click', function() {
            testingPanel.style.display = 'block';
            interactiveTestBtn.style.display = 'none';
            // Focus on input
            setTimeout(() => {
                testChatInput.focus();
            }, 100);
        });
    }
    
    if (testChatInput) {
        // Handle Enter key
        testChatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTestMessage();
            }
        });
    }
    
    if (sendTestBtn) {
        sendTestBtn.addEventListener('click', sendTestMessage);
    }
    
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            const chatHistory = document.getElementById('testChatHistory');
            chatHistory.innerHTML = '<div class="system-message">💡 历史已清空，继续测试...</div>';
        });
    }
    
    if (finishTestBtn) {
        finishTestBtn.addEventListener('click', function() {
            // Trigger download
            const downloadBtn = document.getElementById('downloadBtn');
            if (downloadBtn) {
                downloadBtn.click();
            }
        });
    }

    // Batch testing event listeners
    const batchTestBtn = document.getElementById('batchTestBtn');
    const batchTestPanel = document.getElementById('batchTestPanel');
    const testDatasetFile = document.getElementById('testDatasetFile');
    const uploadTestDatasetBtn = document.getElementById('uploadTestDatasetBtn');
    const runBatchTestBtn = document.getElementById('runBatchTestBtn');
    
    if (batchTestBtn) {
        batchTestBtn.addEventListener('click', function() {
            batchTestPanel.style.display = 'block';
            batchTestBtn.style.display = 'none';
            interactiveTestBtn.style.display = 'inline-block';
        });
    }
    
    if (uploadTestDatasetBtn) {
        uploadTestDatasetBtn.addEventListener('click', uploadTestDataset);
    }
    
    if (runBatchTestBtn) {
        runBatchTestBtn.addEventListener('click', runBatchTest);
    }
});

function startProgressMonitoring() {
    progressInterval = setInterval(checkProgress, 1000);
}

function checkProgress() {
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            updateProgress(data);
            
            if (data.status === 'completed') {
                clearInterval(progressInterval);
                displayResults(data);
            } else if (data.status === 'failed') {
                clearInterval(progressInterval);
                showError(data.message);
            }
        })
        .catch(error => {
            console.error('Progress check failed:', error);
        });
}

function updateProgress(data) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusMessage = document.getElementById('statusMessage');
    const trainingOutput = document.getElementById('trainingOutput');

    const progress = Math.max(0, Math.min(100, data.progress || 0));
    progressFill.style.width = `${progress}%`;
    progressText.textContent = `${progress}%`;
    
    if (data.message) {
        statusMessage.textContent = data.message;
        statusMessage.className = 'status-message loading';
    }
    
    if (data.output) {
        trainingOutput.textContent = data.output;
        trainingOutput.scrollTop = trainingOutput.scrollHeight;
    }
    
    if (data.status === 'completed' || data.status === 'failed') {
        startTrainingBtn.disabled = false;
    }
}

function displayResults(data) {
    const completedFilesList = document.getElementById('completedFiles');
    completedFilesList.innerHTML = '';
    
    if (data.completed_files && data.completed_files.length > 0) {
        data.completed_files.forEach(file => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">(${file.size})</span>
            `;
            li.className = `file-item file-${file.type}`;
            completedFilesList.appendChild(li);
        });
    } else {
        completedFilesList.innerHTML = '<li>没有找到生成的文件</li>';
    }
    
    document.getElementById('resultsCard').style.display = 'block';
    document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const statusMessage = document.getElementById('statusMessage');
    statusMessage.textContent = `❌ ${message}`;
    statusMessage.className = 'status-message error';
}

function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status-message ${type}`;
}

function sendTestMessage() {
    const input = document.getElementById('testChatInput');
    const text = input.value.trim();
    
    if (!text) return;
    
    // Check for exit command
    if (text.toLowerCase() === 'exit') {
        addSystemMessage('测试已结束，您可以下载模型或继续测试');
        input.value = '';
        return;
    }
    
    // Add user message to chat
    addUserMessage(text);
    
    // Clear input and show loading
    input.value = '';
    const loadingId = addLoadingMessage('正在分析...');
    
    // Disable input during processing
    input.disabled = true;
    document.getElementById('sendTestBtn').disabled = true;
    
    // Send test request
    fetch('/test_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_id: currentModelId,
            text: text
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        if (data.success) {
            addBotMessage(text, data.results);
        } else {
            addSystemMessage('❌ 测试失败: ' + data.message);
        }
    })
    .catch(error => {
        removeLoadingMessage(loadingId);
        addSystemMessage('❌ 测试请求失败: ' + error.message);
    })
    .finally(() => {
        // Re-enable input
        input.disabled = false;
        document.getElementById('sendTestBtn').disabled = false;
        input.focus();
    });
}

function addUserMessage(text) {
    const chatHistory = document.getElementById('testChatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'user-message';
    messageDiv.textContent = text;
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(inputText, results) {
    const chatHistory = document.getElementById('testChatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    
    let html = `<div class="result-header">📊 分析结果</div>`;
    
    // Display original model results
    if (results.original) {
        const pred = results.original.prediction;
        const label = results.original.label;
        const probs = results.original.probabilities;
        
        html += `
            <div class="model-result">
                <strong>原始模型:</strong>
                <div class="prediction">${pred} (${label})</div>
                <div class="probabilities">${
                    Object.entries(probs).map(([key, value]) => 
                        `${key}: ${(value * 100).toFixed(1)}%`
                    ).join(' | ')
                }</div>
            </div>
        `;
    }
    
    // Display quantized model results
    if (results.quantized) {
        const pred = results.quantized.prediction;
        const label = results.quantized.label;
        const probs = results.quantized.probabilities;
        
        html += `
            <div class="model-result quantized">
                <strong>量化模型:</strong>
                <div class="prediction">${pred} (${label})</div>
                <div class="probabilities">${
                    Object.entries(probs).map(([key, value]) => 
                        `${key}: ${(value * 100).toFixed(1)}%`
                    ).join(' | ')
                }</div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = html;
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

function addSystemMessage(message) {
    const chatHistory = document.getElementById('testChatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'system-message';
    messageDiv.textContent = message;
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

function addLoadingMessage(message) {
    const chatHistory = document.getElementById('testChatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'loading-message';
    messageDiv.textContent = message;
    const loadingId = 'loading-' + Date.now();
    messageDiv.id = loadingId;
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
    return loadingId;
}

function removeLoadingMessage(loadingId) {
    const loadingMsg = document.getElementById(loadingId);
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

function scrollToBottom() {
    const chatHistory = document.getElementById('testChatHistory');
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function uploadTestDataset() {
    const file = testDatasetFile.files[0];
    if (!file) {
        showStatus('testUploadStatus', '请选择CSV文件', 'error');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        showStatus('testUploadStatus', '请选择CSV格式文件', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('test_dataset', file);
    
    showStatus('testUploadStatus', '上传中...', 'loading');
    uploadTestDatasetBtn.disabled = true;
    
    fetch('/upload_test_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus('testUploadStatus', '✅ 测试集上传成功!', 'success');
            document.getElementById('batchTestControls').style.display = 'block';
        } else {
            showStatus('testUploadStatus', '❌ 上传失败: ' + data.message, 'error');
            uploadTestDatasetBtn.disabled = false;
        }
    })
    .catch(error => {
        showStatus('testUploadStatus', '❌ 上传失败: ' + error.message, 'error');
        uploadTestDatasetBtn.disabled = false;
    });
}

function runBatchTest() {
    const progressContainer = document.getElementById('batchTestProgress');
    const progressBar = document.getElementById('batchTestProgressBar');
    const statusText = document.getElementById('batchTestStatus');
    const resultsContainer = document.getElementById('batchTestResults');
    
    // Show progress
    progressContainer.style.display = 'block';
    resultsContainer.style.display = 'none';
    runBatchTestBtn.disabled = true;
    
    // Start batch testing
    fetch('/batch_test', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model_id: currentModelId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Start monitoring progress
            monitorBatchTestProgress();
        } else {
            showError('批量测试启动失败: ' + data.message);
            progressContainer.style.display = 'none';
            runBatchTestBtn.disabled = false;
        }
    })
    .catch(error => {
        showError('批量测试请求失败: ' + error.message);
        progressContainer.style.display = 'none';
        runBatchTestBtn.disabled = false;
    });
}

function monitorBatchTestProgress() {
    const progressBar = document.getElementById('batchTestProgressBar');
    const statusText = document.getElementById('batchTestStatus');
    
    const checkProgress = () => {
        fetch('/batch_test_progress')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'running') {
                const progress = data.progress || 0;
                progressBar.style.width = progress + '%';
                statusText.textContent = data.message || '测试中...';
                setTimeout(checkProgress, 1000);
            } else if (data.status === 'completed') {
                progressBar.style.width = '100%';
                statusText.textContent = '测试完成!';
                displayBatchTestResults(data.results);
            } else if (data.status === 'error') {
                showError('批量测试失败: ' + data.message);
                document.getElementById('batchTestProgress').style.display = 'none';
                runBatchTestBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Progress check failed:', error);
            setTimeout(checkProgress, 2000);
        });
    };
    
    checkProgress();
}

function displayBatchTestResults(results) {
    const resultsContainer = document.getElementById('batchTestResults');
    const metricsDisplay = document.getElementById('metricsDisplay');
    const errorsDisplay = document.getElementById('errorsDisplay');
    
    // Clear previous results
    metricsDisplay.innerHTML = '';
    errorsDisplay.innerHTML = '';
    
    // Display metrics for each model
    Object.keys(results).forEach(modelType => {
        if (results[modelType].error) {
            metricsDisplay.innerHTML += `
                <div class="model-metrics">
                    <h6>${modelType.charAt(0).toUpperCase() + modelType.slice(1)} Model</h6>
                    <div class="error-message">❌ ${results[modelType].error}</div>
                </div>
            `;
            return;
        }
        
        const metrics = results[modelType];
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-metrics';
        
        modelDiv.innerHTML = `
            <h6>${modelType.charAt(0).toUpperCase() + modelType.slice(1)} Model</h6>
            <div class="metric-item">
                <span class="metric-label">准确率:</span>
                <span class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">精确率:</span>
                <span class="metric-value">${(metrics.precision * 100).toFixed(2)}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">召回率:</span>
                <span class="metric-value">${(metrics.recall * 100).toFixed(2)}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">F1值:</span>
                <span class="metric-value">${(metrics.f1 * 100).toFixed(2)}%</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">测试样本:</span>
                <span class="metric-value">${metrics.total_samples}</span>
            </div>
            <div class="confusion-matrix">
                <strong>混淆矩阵:</strong><br>
                ${metrics.confusion_matrix.map(row => row.join('\t')).join('\n')}
            </div>
        `;
        
        metricsDisplay.appendChild(modelDiv);
    });
    
    // Display error samples for each model separately
    let errorsHtml = '';
    
    Object.keys(results).forEach(modelType => {
        const modelResults = results[modelType];
        if (modelResults && modelResults.errors && modelResults.errors.length > 0) {
            const errors = modelResults.errors;
            const modelName = modelType === 'original' ? '原始模型' : '量化模型';
            
            errorsHtml += `
                <div class="model-errors">
                    <h6>❌ ${modelName}错误样本 (显示前10个，共${errors.length}个错误)</h6>
                    <div class="error-samples">
                        ${errors.slice(0, 10).map((error, index) => `
                            <div class="error-item">
                                <div class="error-text">${index + 1}. ${error.text}</div>
                                <div class="error-labels">
                                    真实: <span class="error-true">${error.true_label_name}</span> | 
                                    预测: <span class="error-pred">${error.predicted_label_name}</span> | 
                                    置信度: ${(error.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <br>
            `;
        }
    });
    
    if (errorsHtml) {
        errorsDisplay.innerHTML = errorsHtml;
    }
    
    // Show results
    resultsContainer.style.display = 'block';
    document.getElementById('batchTestProgress').style.display = 'none';
    runBatchTestBtn.disabled = false;
}
