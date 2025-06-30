from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch

from datetime import datetime

now = datetime.now()
print("start time", now)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# model_name = "google-bert/bert-base-chinese"

#使用本地的模型
model_name = "./models/bert-base-chinese"

# dataset = load_dataset("csv", data_files={"train": ["positive.csv", "negative.csv"], "validation": ["positive_test.csv", "negative_test.csv"]})

#改为自己的数据集地址
dataset = load_dataset("csv", data_files={"train": "./250625/dataset/训练集 (加反例).csv", "validation": "./250625/dataset/验证集 (加反例).csv"})

#分割训练集和验证集
# dataset = load_dataset("csv", data_files="./dataset/六个应用-整合.csv", split="train")
# dataset = dataset.train_test_split(test_size=0.2)
# dataset = load_dataset("json", data_files={"train": "dataset/train.json", "test": "dataset/test.json"}, field="data")
# dataset = DatasetDict({
#     "train": dataset["train"],
#     "validation": dataset["test"]  
# })
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_tokenized_datasets = tokenized_datasets["train"].shuffle(seed=42)#.select(range(100))
validation_tokenized_datasets = tokenized_datasets["validation"].shuffle(seed=42)#.select(range(100))


id2label = {0: "应用商店", 1: "其他"}
label2id = {"应用商店": 0, "其他": 1}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id).to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"  
    )

    #计算AUC
    auc_roc = roc_auc_score(labels, logits[:, 1]) 
    

    return {
        "accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

    # return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./250625/test_trainer_1716",   #改成自己的保存路径
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32, 
    eval_strategy="epoch",
    save_strategy="epoch",          
    load_best_model_at_end=True,    
    metric_for_best_model="f1",  
    greater_is_better=True,      
    num_train_epochs=4,
    learning_rate=1e-5)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
)

trainer = Trainer(
    callbacks=[early_stopping],
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=validation_tokenized_datasets,
    compute_metrics=compute_metrics
    # callbacks=[early_stopping],
)

trainer.train()

now = datetime.now()
print("end time", now)

#改成自己的保存路径
trainer.model.save_pretrained("./250625/trained_model_1716")
tokenizer.save_pretrained("./250625/trained_model_1716")

# trainer.predict(test_datasets)
