import random
import torch
import numpy as np
import json

BUFSIZE = 4096000

def sft_parse_lines(lines, max_len, min_len, separator="|"):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        
        instruction = entry["instruction"].strip()
        output = entry["output"].strip()
        
        if not instruction or not output:
            continue
        
        # 确保 input 存在并进行处理
        input_text = entry.get("input")
        if input_text is not None:
            input_text = input_text.strip()
        else:
            input_text = "" 
        
        # 使用分隔符拼接字段
        full_input = f"{instruction}{separator}{input_text}" if input_text else instruction
        
        # 分割成小块
        full_input = [w for w in full_input]
        output = [w for w in output]
        
        input_sents = chunks(full_input, max_len)
        output_sents = chunks(output, max_len)

        # 只保留长度合适的句子
        if len(input_sents[-1]) < min_len or len(output_sents[-1]) < min_len:
            input_sents = input_sents[:-1]
            output_sents = output_sents[:-1]

        data.extend(zip(input_sents, output_sents))  # 组合输入和输出
    return data


def sft_batchify(data, tknizer):
    truth, inp, msk = [], [], []
    for input_text, output_text in data:
        inp.append(input_text)
        truth.append(output_text)
        msk.append([1 for i in range(len(output_text))])  # 输出的掩码

    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk



def s2t(strs, tknizer):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


def chunks(l, n):
    n = max(1, n)
    return [l[i: i + n] for i in range(0, len(l), n)]


def parse_lines(lines, max_len, min_len):
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = json.loads(line)["text"].strip()
        if not line:
            continue
        line = [w for w in line]
        sents = chunks(line, max_len)
        if len(sents[-1]) < min_len:
            sents = sents[:-1]
        data.extend(sents)
    return data


def ListsToTensor(xs, tknizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if tknizer is not None:
            y = tknizer.token2idx(
                x) + [tknizer.padding_idx] * (max_len - len(x))
        else:
            y = x + [0]*(max_len - len(x))
        ys.append(y)
    return ys


def batchify(data, tknizer):
    truth, inp, msk = [], [], []
    for x in data:
        inp.append(x[:-1])
        truth.append(x[1:])
        msk.append([1 for i in range(len(x) - 1)])

    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return truth, inp, msk


class DataLoader(object):
    def __init__(self, tknizer, filename, batch_size, max_len, min_len, train_type):
        self.batch_size = batch_size
        self.tknizer = tknizer
        self.max_len = max_len
        self.min_len = min_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0
        self.train_type = train_type

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)
        if self.train_type == "pretrain":
            data = parse_lines(lines[:-1], self.max_len, self.min_len)
            random.shuffle(data)

            idx = 0
            while idx < len(data):
                yield batchify(data[idx:idx+self.batch_size], self.tknizer)
                idx += self.batch_size
        else:
            self.epoch_id = 1 
            data = sft_parse_lines(lines[:-1], self.max_len, self.min_len)
            random.shuffle(data)

            idx = 0
            while idx < len(data):
                yield sft_batchify(data[idx:idx+self.batch_size], self.tknizer)
                idx += self.batch_size
