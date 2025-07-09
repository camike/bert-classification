# BERT文本分类C++推理

这个项目使用ONNX Runtime在C++中实现BERT模型的推理，用于文本分类任务。项目支持标准FP32模型和INT8量化模型，可以进行性能对比。

## 目录结构

```
cpp/
├── include/             # 头文件
│   ├── bert_tokenizer.h # BERT分词器实现
│   └── sal_annotations.h # SAL注解（用于MinGW兼容）
├── src/                 # 源代码
│   ├── bert_inference.cpp      # 标准模型推理程序
│   └── bert_inference_int8.cpp # INT8量化模型推理程序
├── model/               # 模型文件
│   ├── model.onnx           # 标准ONNX模型
│   ├── bert_model_quant.onnx # INT8量化ONNX模型
│   └── vocab.txt        # 词表
├── lib/                 # 依赖库
│   └── onnxruntime/     # ONNX Runtime库
├── bin/                 # 输出目录
└── CMakeLists.txt       # CMake配置文件
```

## 环境要求

- C++17或更高版本
- CMake 3.12或更高版本
- MinGW-w64 (Windows) 或 GCC (Linux)
- ONNX Runtime 1.16.0或更高版本（已包含在lib目录中）

## 编译指南

### Windows (使用MinGW)

```powershell
# 创建构建目录
cd cpp
mkdir build
cd build

# 生成Makefile (PowerShell)
cmake .. -G "MinGW Makefiles"

# 编译
mingw32-make
```

## 运行程序

编译完成后，可执行文件将位于`build/bin`目录中。

### 运行标准FP32模型推理

```powershell
# Windows
cd build/bin
./bert_inference.exe

```

### 运行INT8量化模型推理

```powershell
# Windows
cd build/bin
./bert_inference_int8.exe

```

## 使用方法

1. 运行程序后，会显示模型加载信息和当前内存占用
2. 在提示符处输入要分类的文本
3. 程序会显示分类结果、置信度分数、推理时间、内存占用等信息
4. 输入`exit`退出程序


## 功能说明

程序支持以下功能：

1. 加载BERT模型和词表
2. 实现完整的BERT分词器，支持中文分词
3. 文本编码和推理
4. 分类输出
5. 内存占用监控

## 分类类别

目前支持的分类类别：

- 其他
- 爱奇艺
- 飞书
- 鲁大师 

## 内存监控

程序会显示以下内存相关信息：

- 模型加载后的总内存占用
- 推理过程中的总内存占用
- 模型文件大小

## 注意事项

1. 确保模型文件和词表文件位于正确的位置
2. INT8量化模型需要提前准备好，可以使用Python脚本生成
3. 对于大文本输入，推理时间和内存占用会相应增加