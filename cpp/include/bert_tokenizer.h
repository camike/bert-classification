#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <codecvt>
#include <locale>
#include <regex>
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#endif

// 处理Unicode字符和转换的工具函数
class UnicodeUtils {
public:
    // UTF-8转宽字符 - 更可靠的实现
    static std::wstring utf8_to_wstring(const std::string& str) {
        if (str.empty()) {
            return std::wstring();
        }
        
#ifdef _WIN32
        // 使用Windows API进行转换
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
        if (size_needed <= 0) {
            std::cerr << "UTF-8转换错误: 无法计算所需大小" << std::endl;
            return L"";
        }
        
        std::wstring result(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &result[0], size_needed);
        return result;
#else
        // 非Windows系统使用标准方法
        try {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.from_bytes(str);
        } catch (const std::exception& e) {
            std::cerr << "UTF-8转换错误: " << e.what() << std::endl;
            return L"";
        }
#endif
    }
    
    // 宽字符转UTF-8 - 更可靠的实现
    static std::string wstring_to_utf8(const std::wstring& str) {
        if (str.empty()) {
            return std::string();
        }
        
#ifdef _WIN32
        // 使用Windows API进行转换
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0, NULL, NULL);
        if (size_needed <= 0) {
            std::cerr << "宽字符转换错误: 无法计算所需大小" << std::endl;
            return "";
        }
        
        std::string result(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int)str.size(), &result[0], size_needed, NULL, NULL);
        return result;
#else
        // 非Windows系统使用标准方法
        try {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.to_bytes(str);
        } catch (const std::exception& e) {
            std::cerr << "宽字符转换错误: " << e.what() << std::endl;
            return "";
        }
#endif
    }
    
    // 检查字符是否是空白字符
    static bool is_whitespace(wchar_t c) {
        return c == L' ' || c == L'\t' || c == L'\n' || c == L'\r';
    }
    
    // 检查字符是否是控制字符
    static bool is_control(wchar_t c) {
        if (c >= 0 && c <= 31 || c == 127) {
            return true;
        }
        return false;
    }
    
    // 检查字符是否是中文字符
    static bool is_chinese_char(wchar_t c) {
        if ((c >= 0x4E00 && c <= 0x9FFF) ||  // CJK统一汉字 
            (c >= 0x3400 && c <= 0x4DBF) ||  // CJK扩展A
            (c >= 0x20000 && c <= 0x2A6DF) ||  // CJK扩展B
            (c >= 0x2A700 && c <= 0x2B73F) ||  // CJK扩展C
            (c >= 0x2B740 && c <= 0x2B81F) ||  // CJK扩展D
            (c >= 0x2B820 && c <= 0x2CEAF) ||  // CJK扩展E
            (c >= 0xF900 && c <= 0xFAFF) ||   // CJK兼容
            (c >= 0x2F800 && c <= 0x2FA1F)) {  // CJK兼容扩展
            return true;
        }
        return false;
    }
    
    // 检查字符是否是标点符号
    static bool is_punctuation(wchar_t c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
            (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        // 中文标点
        if ((c >= 0x3000 && c <= 0x303F) ||  // CJK标点
            (c >= 0xFF00 && c <= 0xFF0F) ||  // 全角ASCII标点
            (c >= 0xFF1A && c <= 0xFF20) ||  // 全角ASCII标点
            (c >= 0xFF3B && c <= 0xFF40) ||  // 全角ASCII标点
            (c >= 0xFF5B && c <= 0xFF64) ||  // 全角ASCII标点
            (c >= 0xFE30 && c <= 0xFE4F)) {  // CJK兼容形式
            return true;
        }
        return false;
    }
};

// 改进版BERT分词器，更接近HuggingFace实现
class BertTokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> ids_to_tokens;
    
    // 特殊token
    std::string unk_token = "[UNK]";
    std::string cls_token = "[CLS]";
    std::string sep_token = "[SEP]";
    std::string pad_token = "[PAD]";
    std::string mask_token = "[MASK]";
    
    // 配置参数
    bool do_lower_case = false;
    bool tokenize_chinese_chars = true;
    int max_len = 512;

    // 清理文本 - 移除控制字符，规范化空白
    std::string clean_text(const std::string& text) {
        std::cout << "原始输入: '" << text << "'" << std::endl;
        
        std::wstring wtext = UnicodeUtils::utf8_to_wstring(text);
        std::cout << "转换后宽字符长度: " << wtext.size() << std::endl;
        
        std::wstring cleaned;
        
        for (wchar_t c : wtext) {
            if (UnicodeUtils::is_control(c)) {
                continue;  // 跳过控制字符
            }
            if (UnicodeUtils::is_whitespace(c)) {
                cleaned += L' ';  // 规范化为空格
            } else {
                cleaned += c;
            }
        }
        
        std::string result = UnicodeUtils::wstring_to_utf8(cleaned);
        std::cout << "清理后文本: '" << result << "'" << std::endl;
        return result;
    }

    // 基本分词 - 更接近HuggingFace的BasicTokenizer
    std::vector<std::string> basic_tokenize(const std::string& text) {
        // 1. 清理文本
        std::string cleaned_text = clean_text(text);
        
        // 2. 转换为宽字符进行处理
        std::wstring wtext = UnicodeUtils::utf8_to_wstring(cleaned_text);
        std::cout << "基本分词 - 宽字符长度: " << wtext.size() << std::endl;
        
        // 3. 在中文字符周围添加空格
        std::wstring spaced_text;
        for (wchar_t c : wtext) {
            if (UnicodeUtils::is_chinese_char(c)) {
                std::cout << "检测到中文字符: " << static_cast<int>(c) << std::endl;
                spaced_text += L' ';
                spaced_text += c;
                spaced_text += L' ';
            } else {
                spaced_text += c;
            }
        }
        
        // 4. 在标点符号周围添加空格
        std::wstring final_text;
        for (wchar_t c : spaced_text) {
            if (UnicodeUtils::is_punctuation(c)) {
                final_text += L' ';
                final_text += c;
                final_text += L' ';
            } else {
                final_text += c;
            }
        }
        
        // 5. 按空白分割
        std::string final_str = UnicodeUtils::wstring_to_utf8(final_text);
        std::cout << "空格分割前文本: '" << final_str << "'" << std::endl;
        
        std::vector<std::string> tokens;
        std::istringstream iss(final_str);
        std::string token;
        
        while (iss >> token) {
            if (!token.empty()) {
                // 6. 可选的小写转换
                if (do_lower_case) {
                    std::transform(token.begin(), token.end(), token.begin(),
                                   [](unsigned char c){ return std::tolower(c); });
                }
                tokens.push_back(token);
            }
        }
        
        std::cout << "基本分词结果数量: " << tokens.size() << std::endl;
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "  Token " << i << ": '" << tokens[i] << "'" << std::endl;
        }
        
        return tokens;
    }

    // WordPiece分词 - 修复版实现，更接近HuggingFace
    std::vector<std::string> wordpiece_tokenize(const std::string& token) {
        if (token.empty()) {
            return {};
        }

        // 先转换为小写（如果需要）
        std::string processed_token = token;
        if (do_lower_case) {
            std::transform(processed_token.begin(), processed_token.end(),
                         processed_token.begin(), ::tolower);
        }

        // 检查整个token是否在词汇表中
        if (vocab.find(processed_token) != vocab.end()) {
            return {processed_token};
        }

        // WordPiece分解
        std::vector<std::string> sub_tokens;
        int start = 0;

        while (start < processed_token.size()) {
            int end = processed_token.size();
            std::string cur_substr;
            bool found = false;

            // 从最长子串开始尝试
            while (start < end) {
                std::string substr = processed_token.substr(start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }

                if (vocab.find(substr) != vocab.end()) {
                    cur_substr = substr;
                    found = true;
                    break;
                }
                end--;
            }

            if (!found) {
                // 如果找不到匹配的子串，返回UNK
                return {unk_token};
            }

            sub_tokens.push_back(cur_substr);
            start = end;
        }

        return sub_tokens;
    }

public:
    BertTokenizer(const std::string& vocab_file, bool do_lower_case = false) 
        : do_lower_case(do_lower_case) {
        // 加载词表
        std::ifstream file(vocab_file);
        if (!file.is_open()) {
            std::cerr << "无法打开词表文件: " << vocab_file << std::endl;
            return;
        }

        std::string token;
        int idx = 0;
        while (std::getline(file, token)) {
            // 移除换行符
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }
            if (!token.empty() && token.back() == '\n') {
                token.pop_back();
            }
            
            vocab[token] = idx;
            ids_to_tokens.push_back(token);
            idx++;
        }
        
        std::cout << "改进版词表加载完成，大小: " << vocab.size() << std::endl;
    }

    // 分词主函数
    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        
        // 调试信息
        std::cout << "开始分词，输入: '" << text << "'" << std::endl;
        
        // 直接检查是否包含中文字符
        bool has_chinese = false;
        for (unsigned char c : text) {
            if ((c & 0x80) != 0) {  // 非ASCII字符
                has_chinese = true;
                break;
            }
        }
        
        if (has_chinese) {
            std::cout << "检测到中文字符，使用单字符分词" << std::endl;
            // 对中文文本进行单字符分词
            std::string cleaned = clean_text(text);
            std::string current;
            
            for (size_t i = 0; i < cleaned.size();) {
                unsigned char c = cleaned[i];
                if ((c & 0x80) == 0) {  // ASCII字符
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                        if (!current.empty()) {
                            tokens.push_back(current);
                            current.clear();
                        }
                    } else {
                        current += c;
                    }
                    i++;
                } else {
                    // 处理UTF-8多字节字符
                    if (!current.empty()) {
                        tokens.push_back(current);
                        current.clear();
                    }
                    
                    // 获取完整的UTF-8字符
                    std::string utf8_char;
                    if ((c & 0xE0) == 0xC0) {  // 2字节UTF-8
                        if (i + 1 < cleaned.size()) {
                            utf8_char = cleaned.substr(i, 2);
                            i += 2;
                        } else {
                            utf8_char = cleaned.substr(i, 1);
                            i += 1;
                        }
                    } else if ((c & 0xF0) == 0xE0) {  // 3字节UTF-8
                        if (i + 2 < cleaned.size()) {
                            utf8_char = cleaned.substr(i, 3);
                            i += 3;
                        } else {
                            utf8_char = cleaned.substr(i, 1);
                            i += 1;
                        }
                    } else if ((c & 0xF8) == 0xF0) {  // 4字节UTF-8
                        if (i + 3 < cleaned.size()) {
                            utf8_char = cleaned.substr(i, 4);
                            i += 4;
                        } else {
                            utf8_char = cleaned.substr(i, 1);
                            i += 1;
                        }
                    } else {
                        utf8_char = cleaned.substr(i, 1);
                        i += 1;
                    }
                    
                    tokens.push_back(utf8_char);
                }
            }
            
            if (!current.empty()) {
                tokens.push_back(current);
            }
        } else {
            // 1. 基本分词
            std::vector<std::string> basic_tokens = basic_tokenize(text);
            
            // 2. 对每个基本token进行WordPiece分词
            for (const auto& token : basic_tokens) {
                std::vector<std::string> sub_tokens = wordpiece_tokenize(token);
                tokens.insert(tokens.end(), sub_tokens.begin(), sub_tokens.end());
            }
        }
        
        // 打印分词结果
        std::cout << "最终分词结果数量: " << tokens.size() << std::endl;
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "  最终Token " << i << ": '" << tokens[i] << "'" << std::endl;
        }
        
        return tokens;
    }

    // 将文本转换为模型输入
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> 
    encode(const std::string& text, int max_length = 128, bool pad_to_max_length = true) {
        std::vector<std::string> tokens = tokenize(text);
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> token_type_ids;
        
        // 添加[CLS]
        input_ids.push_back(vocab[cls_token]);
        
        // 添加文本token
        for (const auto& token : tokens) {
            if (input_ids.size() >= max_length - 1) break; // 保留[SEP]的位置
            
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                input_ids.push_back(it->second);
            } else {
                input_ids.push_back(vocab[unk_token]);
            }
        }
        
        // 添加[SEP]
        input_ids.push_back(vocab[sep_token]);
        
        // 创建attention_mask和token_type_ids
        attention_mask.resize(input_ids.size(), 1);
        token_type_ids.resize(input_ids.size(), 0);
        
        // 如果需要，填充到最大长度
        if (pad_to_max_length) {
            while (input_ids.size() < max_length) {
                input_ids.push_back(vocab[pad_token]);
                attention_mask.push_back(0);
                token_type_ids.push_back(0);
            }
        }
        
        return {input_ids, attention_mask, token_type_ids};
    }
    
    // 获取词汇表大小
    size_t vocab_size() const {
        return vocab.size();
    }
    
    // 检查token是否存在
    bool has_token(const std::string& token) const {
        return vocab.find(token) != vocab.end();
    }
};
