 
import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME 

model_dir = "output/"
num_labels = 4

model = TinyBertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)

model.eval()
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

id2label = {0: "其他", 1: "爱奇艺", 2: "飞书", 3: "鲁大师"}



def tokenize(text):
    tokens_a = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    input_ids = torch.tensor(input_ids).reshape(1, -1)
    token_type_ids = torch.tensor(segment_ids).reshape(1, -1)
    attention_mask = torch.tensor(input_mask).reshape(1, -1)
    return (input_ids, token_type_ids, attention_mask)

while True:
    text = input("输入 exit 退出: ").strip()
    if text.lower() == "exit":
        exit(0)
    
    inputs = tokenize(text)

    with torch.no_grad():
        logits = model(*inputs)[0]

    predicted_class_id = logits.argmax().item()

    print(id2label[predicted_class_id])