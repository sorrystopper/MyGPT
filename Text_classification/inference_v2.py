import torch
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

# 新的 id2label 和 label2id
id2label = {
    0: '娱乐', 1: '故事', 2: '三农', 3: '房产', 4: '股票', 5: '游戏', 6: '旅游',
    7: '汽车', 8: '体育', 9: '国际', 10: '军事', 11: '文化', 12: '财经', 13: '教育', 14: '科技'
}
label2id={'娱乐': 0, '故事': 1, '三农': 2, '房产': 3, '股票': 4, '游戏': 5, '旅游': 6, 
           '汽车': 7, '体育': 8, '国际': 9, '军事': 10, '文化': 11, '财经': 12, '教育': 13, '科技': 14}

# 加载数据函数
def load_data_to_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                label, text = line.split('_!_')  # 按分隔符拆分
                data.append({'label': label2id[label], 'text': text})
    return Dataset.from_list(data)

# 加载数据集
test_file = "./toutiao_cat_data/test_data.txt"
data_test = load_data_to_dataset(test_file)

# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("./full_model_tokenizer")
inference_model = AutoModelForSequenceClassification.from_pretrained("./full_model").to('cuda')

# 数据集的 Padding 工具
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# 分类函数
def classify_batch(data):
    predictions = []
    labels = []
    for example in tqdm(data, desc="Test:"):
        text = example["text"]
        label = example["label"]
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            output = inference_model(**inputs)
            pred = output.logits.argmax(dim=-1).item()
        predictions.append(pred)
        labels.append(label)
    return predictions, labels

# 计算测试集预测结果
y_pred, y_true = classify_batch(data_test)

# 计算分类报告
label_set = list(id2label.values())
report = classification_report(y_true, y_pred, target_names=label_set, zero_division=0, output_dict=True)

# 保存到 CSV 文件
df = pd.DataFrame(report).transpose()
output_file = "classification_report_toutiao.csv"
df.to_csv(output_file)

# 打印详细结果
print("分类结果：")
print(report)
