import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import csv
from datetime import datetime

model_name = "./output/Qwen1.5/checkpoint-4000"
csv_path = "classification_results.csv"

def predict(messages, model, tokenizer):
    device = "cuda:0"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=64)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/xli/models/Qwen2___5-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda:0")

# 加载训练好的 Lora 模型
model = PeftModel.from_pretrained(model, model_id=model_name)

# 加载测试数据
test_file = "/home/xli/skw/MyGPT/Text_classification/toutiao_cat_data/test_data.txt"

texts = []
labels = []
label_set = set()

max_samples = 1000
sample_count = 0

with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        # if sample_count >= max_samples:
        #     break
        category, text = line.strip().split("_!_")
        labels.append(category)
        texts.append(text)
        label_set.add(category)
        # sample_count += 1


label_set = sorted(list(label_set))  # 确保类别有序
label_to_id = {label: idx for idx, label in enumerate(label_set)}

# 初始化预测和真实标签
y_true = []
y_pred = []

print("开始分类任务...")
for text, true_label in tqdm(zip(texts, labels), total=len(texts), desc="分类进度"):
    instruction = "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型。"
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"文本: {text}"}
    ]
    response = predict(messages, model, tokenizer)
    y_true.append(label_to_id[true_label])
    # 假设模型返回的类别标签直接与 `label_set` 中的类别一致
    y_pred.append(label_to_id.get(response.strip(), 0))  # 如果未知类别，设为 -1

# 计算分类指标
report = classification_report(y_true, y_pred, target_names=label_set, zero_division=0,output_dict=True)
import pandas as pd
df = pd.DataFrame(report).transpose()


output_file = "classification_report_4000.csv"
df.to_csv(output_file)
# 打印详细结果
print("分类结果：")
print(report)
