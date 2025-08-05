# 模型微调与优化流水线系统

## 项目简介
本项目提供了一套完整的模型微调、格式转换和量化压缩的自动化流水线系统，支持BERT、TinyBERT、TinyBERT（蒸馏）。

## 运行说明
python server.py 命令可以让用户使用网页版的模型微调
python run_all.py 命令可以让用户在终端进行模型微调 (终端路径需要调到train_conv_quant子文件夹)

## 目录结构说明（主要文件）
train_models/
│
├── server.py                       # 后端服务主程序 （调用run_all.py脚本）
├── templates/                      # HTML模板文件
│   ├── index.html                  # 首页模板
│   ├── pipeline.html               # 微调流水线界面
│   └── result.html                 # 微调结果展示界面
│
├── static/                         # 静态资源
│   ├── script.js                   # 前端交互脚本
│   └── style.css                   # 样式表
│
└── train_conv_quant/               # 核心训练和转换脚本
    ├── run_all.py                  # 全流程自动化脚本 （调用微调/测试脚本）
    │
    ├── train_bert.py               # BERT微调脚本
    ├── train_tinybert.py           # TinyBERT微调脚本
    ├── train_tinybert_distilled.py # TinyBERT蒸馏微调脚本
    │
    ├── conversion.py               # PyTorch转ONNX脚本
    ├── dynamic_quantization.py     # 动态量化脚本
    ├── accuracy_summary.py         # 模型精度评估脚本
    └── batch_testing.py            # 批量测试脚本