
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
import time

text = "文件之家兼容性问题解决方法"
text = "feishu官网下载"
text = "东方财富下载安装"
text = "东方财富"

#修改为本地路径
model_name = "./250625/test_trainer_1716/checkpoint-1125"
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="cpu")
#修改为本地路径
tokenizer = AutoTokenizer.from_pretrained("./250625/trained_model_1716")

while True:
    text = input("输入 exit 退出: ").strip()
    if text.lower() == "exit":
        exit(0)
    
    start_time = time.time()
    # print("start time", now)
    inputs = tokenizer(text, return_tensors="pt").to("cpu")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()

    print(model.config.id2label[predicted_class_id])
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"use {inference_time:.4f} seconds")
