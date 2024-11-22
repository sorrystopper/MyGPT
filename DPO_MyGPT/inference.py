import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy
import time

import sys
import os

sys.path.append('/home/xli/skw/MyGPT')

from mygpt import myGPT
from tokenizer import Tokenizer
from bpe_tokenizer import BPE_Tokenizer
from inference import s2t


def init_model(m_path, device, vocab):
    ckpt = torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    if vocab_type == "char-based":
        lm_vocab = Tokenizer(
            vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    else:
        lm_vocab = BPE_Tokenizer(model_path="./model/m.model")
    lm_model = myGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim,
                     lm_args.num_heads, lm_args.dropout, lm_args.layers)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args


@torch.no_grad()
def top_k_inc(lm_model, lm_vocab, device, s, k, max_len):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred, incremental_state = lm_model.work_incremental(
            x, incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples=1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        # ipdb.set_trace()
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == '<eos>':
                res.append(sent)
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(device)
        bidx = torch.BoolTensor(bidx).to(device)
        incremental_state["bidx"] = bidx
    res += s_
    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>")[1], x, probs
    else:
        return r, x, probs


if __name__ == "__main__":
    val_type = "sft"
    vocab_type = "char-based"
    device = 1
    print("loading raw_model")
    m_path = "/home/xli/skw/MyGPT/ckpt/epoch1_batch_9999"
    v_path = "/home/xli/skw/MyGPT/model/vocab.txt"
    raw_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")
    print("loading dpo_model")
    ckpt = torch.load("/home/xli/skw/MyGPT/DPO_MyGPT/ckpt/Qwen1.5-0.5B-Chat/dpo_model", map_location='cpu')
    dpo_vocab = Tokenizer(v_path, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    dpo_model = myGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim,
                      lm_args.num_heads, lm_args.dropout, lm_args.layers)
    dpo_model.load_state_dict(ckpt['model'])
    dpo_model = dpo_model.to(device)
    dpo_model.eval()
    print("done.")

    instruction = "以下是描述任务的说明。编写适当地完成请求的响应。"
    query = "哪一个富含蛋白质，床还是墙？"
    messages = [
        {"role": "system", "content": "你是一个非常有帮助和智能的助手。"},
        {"role": "instrution", "content": instruction},
        {"role": "user", "content": query},
    ]
    text = ""
    for message in messages:
        role = message['role']
        content = message['content']
        # 每条消息的格式：<|im_start|>role\ncontent<|im_end|>\n
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    raw_response, _, _ = top_k_inc(raw_model, lm_vocab, device, [[text]], 50, 250)
    dpo_respose, _, _ = top_k_inc(dpo_model, dpo_vocab, device, [[text]], 50, 250)
    print(f"dpo_respose:{dpo_respose}")
    print(f"raw_response:{raw_response}")