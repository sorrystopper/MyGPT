import torch

def generate(model, tokenizer, device, instruction, top_k, top_p, max_new_tokens, temperature):
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        input_ids = tokenizer.encode(instruction)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)  
        
        generated = input_tensor.tolist()
        
        for _ in range(max_new_tokens):
            last_token_tensor = torch.tensor(generated[0][-1]).unsqueeze(0).unsqueeze(0).to(device)  
            outputs, pred_y = model.work(last_token_tensor)  # 获取模型的输出
            logits = outputs[-1, -1, :] / temperature  # 应用温度

            if top_k > 0:
                indices_to_keep = logits.topk(top_k).indices
                logits = torch.full_like(logits, -float('Inf'))
                logits[indices_to_keep] = logits[indices_to_keep]
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_keep = cumulative_probs <= top_p
                indices_to_keep = sorted_indices[sorted_indices_to_keep]
                logits = torch.full_like(logits, -float('Inf'))
                logits[indices_to_keep] = logits[indices_to_keep]

            # 进行采样
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()  # 采样一个 token
            
            generated[0].append(next_token)  # 添加到生成序列

            # 如果遇到结束标记，提前停止
            if next_token == '<eos>':
                break

    response = tokenizer.decode(generated[0][len(input_ids):])  # 解码生成的序列
    logits = outputs  # 返回logits
    return input_tensor, response, logits
