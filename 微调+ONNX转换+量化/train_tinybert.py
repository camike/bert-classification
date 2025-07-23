################################################
# 本文件用于直接微调TinyBert模型（有20%验证集）
#
# 功能说明：
# - 输入: TinyBert模型路径；数据集（自动分割训练/验证）
# - 输出: 微调后的TinyBert模型
#
# 需要修改的配置项（共4处需要修改，已在代码中用注释"x"标记）：
# 1. TinyBert模型路径：第26行
# 2. 数据集路径：第29行
# 3. test trainer文件保存路径：第65行
# 4. 微调模型文件保存路径：第105行
################################################



import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from tqdm.auto import tqdm
import os

# 1. Set the correct model path
model_path = "TinyBERT/TinyBERT_model" # 修改为TinyBert模型路径（1）

# 2. Load your dataset
df = pd.read_csv('dataset.csv') # 修改为数据集路径（2）
texts = df['text'].tolist()
labels = df['label'].tolist()

# 3. Verify 4 classes
num_classes = len(set(labels))
assert num_classes == 4, f"Expected 4 classes, found {num_classes}"
print(f"✓ Dataset verified with {num_classes} classes")

# 4. Load tokenizer and model
print(f"Loading model from: {model_path}")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=4,
    ignore_mismatched_sizes=True
)
print("✓ Model loaded successfully")

# 5. Prepare datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

print("Tokenizing data...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 6. Updated Training setup with correct parameter names
training_args = TrainingArguments(
    output_dir="./tinybert_training_results", # 修改为test trainer文件保存路径（3）
    eval_strategy="epoch", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    num_train_epochs=4,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 7. Train with progress tracking
print("Starting training...")
for epoch in range(training_args.num_train_epochs):
    print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Validation results:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

# 8. Save the fine-tuned model
output_dir = "tb001" # 修改为微调模型文件保存路径（4）
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✓ Training complete! Model saved to {output_dir}")