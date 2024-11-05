import torch
from inference import greedy, top_k_inc,top_p_inc

def generate(model, tokenizer, device, instruction, top_k, top_p, max_new_tokens, temperature):
    s = [[w for w in instruction]]
    if top_k > 0:
        r, x, probs = top_k_inc(model, tokenizer,device,s, top_k, max_new_tokens)
    elif top_p < 1.0:
        r, x, probs = top_p_inc(model, tokenizer,device,s, top_k, top_p, max_new_tokens)
    else:
        r, x, probs = greedy(model, tokenizer, device, s, max_new_tokens)
    return x, r, probs[-1]
