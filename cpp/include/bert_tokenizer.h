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

// 处理Unicode字符和转换的工具函数
class UnicodeUtils {
public:
    // UTF-8转宽字符
    static std::wstring utf8_to_wstring(const std::string& str) {
        try {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.from_bytes(str);
        } catch (const std::exception& e) {
            std::cerr << "UTF-8转换错误: " << e.what() << std::endl;
            return L"";
        }
    }
    
    // 宽字符转UTF-8
    static std::string wstring_to_utf8(const std::wstring& str) {
            std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
            return converter.to_bytes(str);
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
        std::wstring wtext = UnicodeUtils::utf8_to_wstring(text);
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
        
        return UnicodeUtils::wstring_to_utf8(cleaned);
    }

    // 基本分词 - 更接近HuggingFace的BasicTokenizer
    std::vector<std::string> basic_tokenize(const std::string& text) {
        // 1. 清理文本
        std::string cleaned_text = clean_text(text);
        
        // 2. 转换为宽字符进行处理
        std::wstring wtext = UnicodeUtils::utf8_to_wstring(cleaned_text);
        
        // 3. 在中文字符周围添加空格
        std::wstring spaced_text;
        for (wchar_t c : wtext) {
            if (UnicodeUtils::is_chinese_char(c)) {
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
        
        // 1. 基本分词
        std::vector<std::string> basic_tokens = basic_tokenize(text);
        
        // 2. 对每个基本token进行WordPiece分词
        for (const auto& token : basic_tokens) {
            std::vector<std::string> sub_tokens = wordpiece_tokenize(token);
            tokens.insert(tokens.end(), sub_tokens.begin(), sub_tokens.end());
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
