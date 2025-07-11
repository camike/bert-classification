#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <numeric>
#include "../include/sal_annotations.h"
#include "../include/bert_tokenizer.h"
#include <onnxruntime_cxx_api.h>

// Windows系统下，添加内存监控所需的头文件
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Linux系统下，添加内存监控所需的头文件
#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

// CSV行数据结构
struct CSVRow {
    int label;
    std::string text;
};

// 获取当前进程的内存使用情况（单位：KB）
size_t getCurrentMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024; // 转换为KB
    }
    return 0;
#elif defined(__linux__)
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) {
        return 0;
    }
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE) / 1024;
#else
    return 0; // 不支持的平台
#endif
}

// 获取内存增量（在推理前后测量内存差异）
size_t getMemoryDelta(std::function<void()> func) {
    // 强制垃圾回收
    for (int i = 0; i < 3; i++) {
        std::vector<char> tmp(1024 * 1024, 0); // 分配1MB
    }
    
    // 测量前的内存
    size_t before = getCurrentMemoryUsage();
    
    // 执行函数
    func();
    
    // 测量后的内存
    size_t after = getCurrentMemoryUsage();
    
    // 返回差值（如果为负，则返回0）
    return (after > before) ? (after - before) : 0;
}

// 模型标签映射 - 使用与之前相同的标签
const std::vector<std::string> id2label = {"其他", "爱奇艺", "飞书", "鲁大师"};

// 获取文件大小（KB）
size_t getFileSize(const std::string& filepath) {
    try {
        return std::filesystem::file_size(filepath) / 1024;
    } catch (const std::exception& e) {
        return 0;
    }
}

// 解析CSV文件
std::vector<CSVRow> parseCSV(const std::string& filepath) {
    std::vector<CSVRow> rows;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "无法打开CSV文件: " << filepath << std::endl;
        return rows;
    }
    
    std::string line;
    // 跳过标题行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label_str, text;
        
        // 解析标签
        if (!std::getline(ss, label_str, ',')) {
            continue;
        }
        
        // 解析文本（可能包含逗号）
        if (!std::getline(ss, text)) {
            continue;
        }
        
        // 处理可能的引号
        if (!text.empty() && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.length() - 2);
        }
        
        try {
            int label = std::stoi(label_str);
            rows.push_back({label, text});
        } catch (const std::exception& e) {
            std::cerr << "解析标签失败: " << label_str << std::endl;
        }
    }
    
    return rows;
}

// 单次推理函数
int performInference(Ort::Session& session, BertTokenizer& tokenizer, const std::string& text, 
                    double& inference_time, size_t& memory_usage) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 创建一个lambda函数来执行推理，以便测量内存使用
    int predicted_class = 0;
    
    try {
        memory_usage = getMemoryDelta([&]() {
            // 对文本进行编码
            auto [input_ids, attention_mask, token_type_ids] = tokenizer.encode(text, 128, false);
            
            // 准备输入张量
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
            
            // 创建内存信息和输入张量
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()
            );
            
            Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()
            );
            
            Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, token_type_ids.data(), token_type_ids.size(), input_shape.data(), input_shape.size()
            );
            
            // 构建输入容器
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_ids_tensor));
            input_tensors.push_back(std::move(attention_mask_tensor));
            input_tensors.push_back(std::move(token_type_ids_tensor));
            
            // 输入输出名称 - 根据模型结构设置
            const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
            const char* output_names[] = {"logits"};  // 输出节点名称，lmb.onnx使用logits而不是output

            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names,
                1
            );
            
            // 处理输出
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            
            // 找到最大值的索引（即预测的类别）
            predicted_class = 0;
            float max_score = output_data[0];
            
            for (size_t i = 0; i < output_size; ++i) {
                if (output_data[i] > max_score) {
                    max_score = output_data[i];
                    predicted_class = static_cast<int>(i);
                }
            }
        });
    } catch (const std::exception& e) {
        std::cerr << "推理过程中出错: " << e.what() << std::endl;
        throw; // 重新抛出异常，让调用者处理
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    inference_time = duration.count();
    
    return predicted_class;
}

// 主函数
int main() {
    try {
        // 设置控制台为UTF-8编码 (不使用system命令，避免清屏)
        #ifdef _WIN32
        SetConsoleOutputCP(65001);  // 使用Windows API直接设置代码页
        SetConsoleCP(65001);        // 设置输入代码页为UTF-8
        #endif
        
        std::cout << "=== LMB模型推理程序（CSV测试版）===" << std::endl;
        
        // 检查模型文件
        std::string model_path = "model/lmb.onnx";
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "错误: 未找到LMB模型文件! 尝试使用其他路径..." << std::endl;
            
            // 尝试其他可能的路径
            model_path = "../model/lmb.onnx";
            if (!std::filesystem::exists(model_path)) {
                model_path = "../../cpp/model/lmb.onnx";
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "错误: 无法找到LMB模型文件!" << std::endl;
                    return 1;
                }
            }
        }
        
        // 获取模型大小
        size_t model_size = getFileSize(model_path);
        
        // 初始化ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lmb-inference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        std::cout << "正在加载模型和分词器..." << std::endl;
        std::cout << "模型路径: " << model_path << std::endl;
        
        // 加载模型
        #ifdef _WIN32
            Ort::Session session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
        #else
            Ort::Session session(env, model_path.c_str(), session_options);
        #endif
        
        // 加载分词器 - 寻找词汇表文件
        std::string vocab_path = "model/vocab.txt";
        if (!std::filesystem::exists(vocab_path)) {
            std::cerr << "警告: 未找到词汇表文件! 尝试使用其他路径..." << std::endl;
            
            // 尝试其他可能的路径
            vocab_path = "../model/vocab.txt";
            if (!std::filesystem::exists(vocab_path)) {
                vocab_path = "../../cpp/model/vocab.txt";
                if (!std::filesystem::exists(vocab_path)) {
                    std::cerr << "错误: 无法找到词汇表文件!" << std::endl;
                    return 1;
                }
            }
        }
        
        std::cout << "词汇表路径: " << vocab_path << std::endl;
        BertTokenizer tokenizer(vocab_path, true);  // do_lower_case=true，与Python版本一致
        
        std::cout << "模型加载完成，模型大小: " << model_size << " KB" << std::endl;
        std::cout << "当前总内存占用: " << getCurrentMemoryUsage() << " KB" << std::endl;
        std::cout << "------------------------------------" << std::endl;
        
        // 主菜单
        while (true) {
            std::cout << "\n请选择操作模式:" << std::endl;
            std::cout << "1. 单条文本推理" << std::endl;
            std::cout << "2. CSV文件批量测试" << std::endl;
            std::cout << "3. 退出程序" << std::endl;
            std::cout << "请输入选择 (1-3): ";
            
            std::string choice;
            std::getline(std::cin, choice);
            
            if (choice == "1") {
                // 单条文本推理模式
                std::string text;
                std::cout << "请输入文本 (输入'back'返回): ";
                std::getline(std::cin, text);
                
                if (text == "back") continue;
                
                double inference_time = 0.0;
                size_t memory_usage = 0;
                
                try {
                    int predicted_class = performInference(session, tokenizer, text, inference_time, memory_usage);
                    
                    // 输出结果
                    std::cout << "\n=== 推理结果 ===" << std::endl;
                    std::cout << "预测类别: " << id2label[predicted_class] << std::endl;
                    std::cout << "推理时间: " << inference_time << " 毫秒" << std::endl;
                    std::cout << "内存增量: " << memory_usage << " KB" << std::endl;
                    std::cout << "总内存占用: " << getCurrentMemoryUsage() << " KB" << std::endl;
                    std::cout << "------------------------------------" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "推理失败: " << e.what() << std::endl;
                }
                
            } else if (choice == "2") {
                // CSV文件批量测试模式
                std::string csv_path;
                std::cout << "请输入CSV文件路径 (输入'back'返回): ";
                std::getline(std::cin, csv_path);
                
                if (csv_path == "back") continue;
                
                // 解析CSV文件
                std::vector<CSVRow> csv_data = parseCSV(csv_path);
                
                if (csv_data.empty()) {
                    std::cout << "CSV文件为空或解析失败!" << std::endl;
                    continue;
                }
                
                std::cout << "成功加载CSV文件，共" << csv_data.size() << "条数据" << std::endl;
                std::cout << "开始批量测试..." << std::endl;
                
                // 性能统计变量
                std::vector<double> inference_times;
                std::vector<size_t> memory_usages;
                int correct_predictions = 0;
                int processed_count = 0;
                
                // 批量测试
                for (size_t i = 0; i < csv_data.size(); ++i) {
                    const auto& row = csv_data[i];
                    
                    double inference_time = 0.0;
                    size_t memory_usage = 0;
                    int predicted_class = 0;
                    
                    try {
                        predicted_class = performInference(session, tokenizer, row.text, inference_time, memory_usage);
                        
                        // 收集统计数据
                        inference_times.push_back(inference_time);
                        memory_usages.push_back(memory_usage);
                        processed_count++;
                        
                        // 检查预测是否正确
                        if (predicted_class == row.label) {
                            correct_predictions++;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "处理第" << (i+1) << "行时出错: " << e.what() << std::endl;
                        std::cerr << "跳过此行" << std::endl;
                    }
                    
                    // 显示进度
                    if ((i + 1) % 10 == 0 || i + 1 == csv_data.size()) {
                        std::cout << "已处理 " << (i + 1) << "/" << csv_data.size() << " 条数据" << std::endl;
                    }
                }
                
                // 计算统计结果
                double avg_inference_time = inference_times.empty() ? 0.0 : 
                    std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
                double avg_memory_usage = memory_usages.empty() ? 0.0 :
                    std::accumulate(memory_usages.begin(), memory_usages.end(), 0.0) / memory_usages.size();
                double accuracy = processed_count == 0 ? 0.0 :
                    static_cast<double>(correct_predictions) / processed_count * 100.0;
                
                // 输出统计结果
                std::cout << "\n=== 测试结果统计 ===" << std::endl;
                std::cout << "测试样本总数: " << csv_data.size() << std::endl;
                std::cout << "成功处理样本数: " << processed_count << std::endl;
                std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_inference_time << " 毫秒" << std::endl;
                std::cout << "平均内存增量: " << std::fixed << std::setprecision(2) << avg_memory_usage << " KB" << std::endl;
                std::cout << "当前总内存占用: " << getCurrentMemoryUsage() << " KB" << std::endl;
                std::cout << "正确预测数量: " << correct_predictions << "/" << processed_count << std::endl;
                std::cout << "准确率: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
                std::cout << "------------------------------------" << std::endl;
                
                // 保存详细结果到CSV文件
                std::string result_file = "lmb_test_results.csv";
                std::ofstream out_file(result_file);
                
                if (out_file.is_open()) {
                    out_file << "文本,真实标签,预测标签,是否正确,推理时间(ms),内存增量(KB)" << std::endl;
                    
                    for (size_t i = 0; i < csv_data.size(); ++i) {
                        const auto& row = csv_data[i];
                        double inference_time = 0.0;
                        size_t memory_usage = 0;
                        int predicted_class = -1;
                        bool is_correct = false;
                        
                        try {
                            predicted_class = performInference(session, tokenizer, row.text, inference_time, memory_usage);
                            is_correct = (predicted_class == row.label);
                            
                            // 处理文本中的逗号，确保CSV格式正确
                            std::string escaped_text = row.text;
                            if (escaped_text.find(',') != std::string::npos) {
                                escaped_text = "\"" + escaped_text + "\"";
                            }
                            
                            out_file << escaped_text << ","
                                    << row.label << ","
                                    << predicted_class << ","
                                    << (is_correct ? "是" : "否") << ","
                                    << inference_time << ","
                                    << memory_usage << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "保存第" << (i+1) << "行结果时出错: " << e.what() << std::endl;
                            
                            // 仍然保存行，但标记为错误
                            std::string escaped_text = row.text;
                            if (escaped_text.find(',') != std::string::npos) {
                                escaped_text = "\"" + escaped_text + "\"";
                            }
                            
                            out_file << escaped_text << ","
                                    << row.label << ","
                                    << "错误" << ","
                                    << "否" << ","
                                    << "0" << ","
                                    << "0" << std::endl;
                        }
                    }
                    
                    out_file.close();
                    std::cout << "详细结果已保存到文件: " << result_file << std::endl;
                } else {
                    std::cerr << "无法创建结果文件!" << std::endl;
                }
                
                // 输出最终的性能摘要
                std::cout << "\n=== 性能摘要 ===" << std::endl;
                std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_inference_time << " 毫秒" << std::endl;
                std::cout << "平均内存增量: " << std::fixed << std::setprecision(2) << avg_memory_usage << " KB" << std::endl;
                std::cout << "准确率: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
                std::cout << "------------------------------------" << std::endl;
                
            } else if (choice == "3") {
                // 退出程序
                break;
            } else {
                std::cout << "无效的选择，请重新输入" << std::endl;
            }
        }
        
        std::cout << "LMB模型推理程序已退出" << std::endl;
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime错误: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "标准错误: " << e.what() << std::endl;
        return 1;
    }
} 