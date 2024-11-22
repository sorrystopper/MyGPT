from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import torch
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
import swanlab

swanlab_callback = SwanLabCallback(
    project="Qwen2-fintune",
    experiment_name="Qwen1.5-0.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在toutiao_cat_data数据集上微调。",
    config={
        "model": "qwen/Qwen2-1.5B-Instruct",
        "dataset": "huangjintao/zh_cls_fudan-news",
    }
)

id2label = {0: '娱乐', 1: '故事', 2: '三农', 3: '房产', 4: '股票', 5: '游戏', 6: '旅游',
            7: '汽车', 8: '体育', 9: '国际', 10: '军事', 11: '文化', 12: '财经', 13: '教育', 14: '科技'}

label2id = {'娱乐': 0, '故事': 1, '三农': 2, '房产': 3, '股票': 4, '游戏': 5, '旅游': 6,
            '汽车': 7, '体育': 8, '国际': 9, '军事': 10, '文化': 11, '财经': 12, '教育': 13, '科技': 14}


def load_data_to_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                label, text = line.split('_!_')  # 按分隔符拆分
                data.append({'label': label2id[label], 'text': text})
    return Dataset.from_list(data)


# 文件路径
train_file = "./toutiao_cat_data/train_data.txt"
test_file = "./toutiao_cat_data/test_data.txt"
val_file = "./toutiao_cat_data/val_data.txt"

# 加载数据
data_train = load_data_to_dataset(train_file)
data_test = load_data_to_dataset(test_file)
data_val = load_data_to_dataset(val_file)

# 【加载分词器】
tokenizer = AutoTokenizer.from_pretrained("/home/xli/skw/model/Qwen1.5-0.5B-Chat", trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Qwen特性，需要指定一下pad_token_id


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


token_train = data_train.map(tokenize_function, batched=True, num_proc=16)
token_val = data_val.map(tokenize_function, batched=True, num_proc=16)

train_dataset = token_train
eval_dataset = token_val

# 使用Qwen1.5模型
model = AutoModelForSequenceClassification.from_pretrained("/home/xli/skw/model/Qwen1.5-0.5B-Chat", num_labels=15, \
                                                           id2label=id2label, label2id=label2id, device_map="auto", \
                                                           torch_dtype=torch.bfloat16)
model.config.pad_token_id = model.config.eos_token_id  # 这里也要指定一下pad_token_id，不然训练时会报错 "ValueError: Cannot handle batch sizes > 1 if no padding token is defined."

# 【训练参数】
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

training_args = TrainingArguments(
    output_dir="./output/Qwen1.5_v2",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_steps=20,
    num_train_epochs=3,
    save_steps=4000,
    learning_rate=1e-4,
    save_on_each_node=True,
    report_to="none",
    logging_first_step=True
)

from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # 获取每个样本的预测类别
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average="macro")  # 设置为多分类模式
    f1 = f1_score(labels, predictions, average="macro")  # 设置为多分类模式
    return {'accuracy': accuracy, 'recall': recall, 'f1': f1}


def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        # 给数据添加padding弄成batch
        callbacks=[swanlab_callback],
    )


# 【完全微调】
print("【开始训练】")
trainer = get_trainer(model)
trainer.train()

tokenizer.save_pretrained("./full_model_tokenizer")
model.save_pretrained("./full_model")

swanlab.finish()

# 【PEFT-LoRA微调】
# from peft import LoraConfig, get_peft_model

# peft_config = LoraConfig(
#     task_type="SEQ_CLS", #任务类型：分类
#     target_modules=["q_proh","k_proj","v_proj","o_proj"],  # 这个不同的模型需要设置不同的参数，主要看模型中的attention层
#     inference_mode=False, # 关闭推理模式 (即开启训练模式)
#     r=8, # Lora 秩
#     lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
#     lora_dropout=0.1 # Dropout 比例
# )

# peft_model = get_peft_model(model, peft_config) # 加载lora参数peft框架

# print('PEFT参数量：')
# peft_model.print_trainable_parameters()

# print("【开始训练】")
# peft_trainer=get_trainer(peft_model)
# peft_trainer.train()

# tokenizer.save_pretrained("./peft_model_tokenizer")
# peft_model.save_pretrained("./peft_model")

