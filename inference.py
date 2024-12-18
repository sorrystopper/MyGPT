import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy
import time

from mygpt import myGPT
from tokenizer import Tokenizer
from bpe_tokenizer import BPE_Tokenizer
from data import DataLoader, s2t


def mstime(): return int(round(time.time() * 1000))


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


def greedy(lm_model, lm_vocab, device, s, max_len):
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            next_tk.append(lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))

        s_ = []
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
    res += s_

    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>"[1]), x, probs
    else:
        return r, x, probs


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
        import ipdb
        ipdb.set_trace()
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


def top_k_inc_dpo(lm_model, lm_vocab, device, inp, msk, max_len, k=5, train_type=0):
    inp = torch.transpose(inp, 0, 1)
    if inp.size(0) > 1024:
        inp = inp[:1024, :]
    inp = inp.to(device)
    incremental_state = None
    probs, pred, incremental_state = lm_model.work_incremental(
        inp, incremental_state)
    # ipdb.set_trace()
    return msk, inp, probs


def top_p_sampling(logits, k, p):
    ps, idx = torch.topk(logits, k=k)
    for i in range(k):
        if torch.sum(ps[:i]) >= p:
            return ps[:i], idx[:i]
    return ps, idx


def top_p_inc(lm_model, lm_vocab, device, s, k, p, max_len):
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
                ps, idx = top_p_sampling(logits, k, p)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = top_p_sampling(logits, k, p)
                ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples=1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))

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
        return r.splot("<bos>")[1], x, probs
    else:
        return r, x, probs


if __name__ == "__main__":
    val_type = "sft"
    vocab_type = "char-based"
    device = 1
    print("loading...")
    m_path = "./ckpt/epoch1_batch_9999"
    v_path = "./model/vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")

    max_len = 250
    if val_type == "sft":
        qs = [
            "### INST:\n 介绍下南京航空航天大学。\n\n### SYS:\n",
            "### INST:\n 白日依山尽，\n\n### SYS:\n",
            "### INST:\n 已知三个数分别为1，2，3，则它们的平均数是？\n\n### SYS:\n",
            "### INST:\n 小明共有15个苹果，他分别给了3个人2个苹果，然后自己又吃了一个苹果，那么他还剩几个苹果？\n\n### SYS:\n",
            "### INST:\n 根据牛顿第二定理，物体的加速度等于\n\n### SYS:\n",
            "### INST:\n 碳纳米管是一种新型的材料，具有非常独特的电学和光学性质。在过去的几年里，我们对碳纳\n\n### SYS:\n",
            "### INST:\n 下面是一段用python写的快速排序代码:\n\n### SYS:\n",
            "### INST:\n 下面是一个使用 PyTorch 和 Transformer 的示例代码，用于训练一个文本分类模型：import torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset\n\n### SYS:\n"
        ]
    else:
        qs = ["介绍下南京航空航天大学。",
              "Please introduce Nanjing University of Aeronautics and Astronautics",
              "The meaning of life is ",
              "白日依山尽，",
              "君不见，黄河之水天上来，奔流到海不复回。君不见，",
              "秦孝公据崤函之固,拥雍州之地,君臣固守以窥周室,有席卷天下,包举宇内,囊括四海之意,并吞八荒之心。",
              "已知三个数分别为1，2，3，则它们的平均数是"]
    print(qs)
    i = 0
    for q in qs:
        start = mstime()
        i += 1
        s = [[w for w in q]]

        r1, _, _ = greedy(lm_model, lm_vocab, device, s, max_len)

        # r2 = beam_search(lm_model, lm_vocab, device, s, max_len)

        r3, _, _ = top_k_inc(lm_model, lm_vocab, device, s, 5, max_len)

        r4, _, _ = top_k_inc(lm_model, lm_vocab, device, s, 10, max_len)

        r5, _, _ = top_k_inc(lm_model, lm_vocab, device, s, 20, max_len)

        r6, _, _ = top_k_inc(lm_model, lm_vocab, device, s, 50, max_len)

        r7, _, _ = top_k_inc(lm_model, lm_vocab, device, s, 500, max_len)

        r8, _, _ = top_p_inc(lm_model, lm_vocab, device, s, 20, 0.95, max_len)

        print(i)
        print("q: ", q)
        print("greedy: ", r1)
        # print("bm5: ", q+r2)
        print("tk5: ", r3)
        print("tk10: ", r4)
        print("tk20: ", r5)
        print("tk50: ", r6)
        print("tk500: ", r7)
        print("tp0.95: ", r8)
        print(mstime() - start)
