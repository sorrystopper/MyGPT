import torch
from inference import greedy, top_k_inc, top_p_inc, top_k_inc_dpo


def generate(model, tokenizer, device, instruction, top_k, top_p, max_new_tokens, temperature):
    s = [[w for w in instruction]]
    if top_k > 0:
        r, x, probs = top_k_inc(model, tokenizer, device,
                                s, top_k, max_new_tokens)
    elif top_p < 1.0:
        r, x, probs = top_p_inc(model, tokenizer, device,
                                s, top_k, top_p, max_new_tokens)
    else:
        r, x, probs = greedy(model, tokenizer, device, s, max_new_tokens)
    return x, r, probs[-1]

def generate_dpo(model, tokenizer, device, top_k, max_new_tokens, input_ids, attention_mask):
    r, x, probs = top_k_inc_dpo(model, tokenizer, device,input_ids, attention_mask,  max_new_tokens, top_k)
    return x, r, probs.transpose(0, 1)