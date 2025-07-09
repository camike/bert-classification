#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
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

// 模型标签映射
const std::vector<std::string> id2label = {"其他", "爱奇艺", "飞书", "鲁大师"};

// 获取文件大小（KB）
size_t getFileSize(const std::string& filepath) {
    try {
        return std::filesystem::file_size(filepath) / 1024;
    } catch (const std::exception& e) {
        return 0;
    }
}

// 主函数
int main() {
    try {
        // 设置控制台为UTF-8编码 (不使用system命令，避免清屏)
        #ifdef _WIN32
        SetConsoleOutputCP(65001);  // 使用Windows API直接设置代码页
        SetConsoleCP(65001);        // 设置输入代码页为UTF-8
        #endif
        
        std::cout << "=== tinybert量化模型推理程序 ===" << std::endl;
        
        // 检查量化模型文件 - 使用当前目录下的model子目录
        std::string quantized_model_path = "model/model.quant.onnx";
        if (!std::filesystem::exists(quantized_model_path)) {
            std::cerr << "错误: 未找到量化模型文件! 尝试使用其他路径..." << std::endl;
            
            // 尝试其他可能的路径
            quantized_model_path = "../model/model.quant.onnx";
            if (!std::filesystem::exists(quantized_model_path)) {
                quantized_model_path = "../../model.quant.onnx";
                if (!std::filesystem::exists(quantized_model_path)) {
                    std::cerr << "错误: 无法找到量化模型文件!" << std::endl;
                    return 1;
                }
            }
        }
        
        // 获取模型大小
        size_t model_size = getFileSize(quantized_model_path);
        
        // 初始化ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bert-optimized-quant-inference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        std::cout << "正在加载模型和分词器..." << std::endl;
        std::cout << "模型路径: " << quantized_model_path << std::endl;
        
        // 加载量化模型
        #ifdef _WIN32
            Ort::Session session(env, std::wstring(quantized_model_path.begin(), quantized_model_path.end()).c_str(), session_options);
        #else
            Ort::Session session(env, quantized_model_path.c_str(), session_options);
        #endif
        
        // 加载改进版分词器 - 寻找词汇表文件
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
        
        // 主循环
        std::string text;
        while (true) {
            std::cout << "请输入文本 (输入'exit'退出): ";
            std::getline(std::cin, text);
            
            if (text == "exit") break;
            
            // 执行推理
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 对文本进行编码 - 使用与Python版本相同的参数
            // Python版本使用padding=True（动态padding），所以这里不使用pad_to_max_length
            auto [input_ids, attention_mask, token_type_ids] = tokenizer.encode(text, 128, false);
            
            // 打印输入ID（与Python版本类似）
            std::cout << "输入ID: [";
            for (size_t i = 0; i < input_ids.size(); ++i) {
                std::cout << input_ids[i];
                if (i < input_ids.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
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
            
            // 输入输出名称 - 使用正确的输出节点名称
            const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
            const char* output_names[] = {"output"};  // 量化模型的输出节点名称

            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names,
                1
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end_time - start_time;
            
            // 处理输出 - 取第一个输出（与Python版本一致）
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            
            // 打印原始输出（调试用）
            std::cout << "原始输出: [";
            for (size_t i = 0; i < output_size; ++i) {
                std::cout << std::fixed << std::setprecision(6) << output_data[i];
                if (i < output_size - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // 找到最大值的索引（即预测的类别）- 与Python的argmax相同
            int predicted_class = 0;
            float max_score = output_data[0];
            std::vector<float> all_scores(output_data, output_data + output_size);
            
            for (size_t i = 0; i < output_size; ++i) {
                if (output_data[i] > max_score) {
                    max_score = output_data[i];
                    predicted_class = static_cast<int>(i);
                }
            }
            
            // 输出结果 - 类似于Python版本
            std::cout << id2label[predicted_class] << std::endl;
            std::cout << "spent " << duration.count() << std::endl;
            
            // 额外输出更多详细信息
            std::cout << "\n=== 详细信息 ===" << std::endl;
            std::cout << "预测结果: " << id2label[predicted_class] << std::endl;
            std::cout << "推理时间: " << duration.count() << " 毫秒" << std::endl;
            std::cout << "总内存占用: " << getCurrentMemoryUsage() << " KB" << std::endl;
            
            // 显示所有类别的分数
            std::cout << "\n各类别详细分数:" << std::endl;
            for (size_t i = 0; i < id2label.size() && i < all_scores.size(); ++i) {
                std::cout << "  " << id2label[i] << ": " << std::fixed << std::setprecision(6) << all_scores[i];
                if (i == predicted_class) {
                    std::cout << " (最高)";
                }
                std::cout << std::endl;
            }
            std::cout << "------------------------------------" << std::endl;
        }
        
        std::cout << "tinybert量化模型推理程序已退出" << std::endl;
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime错误: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "标准错误: " << e.what() << std::endl;
        return 1;
    }
}
