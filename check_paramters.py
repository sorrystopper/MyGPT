import torch


def count_parameters(ckpt_path):
    # 加载模型检查点
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model']  # 提取模型参数

    total_params = 0
    trainable_params = 0

    # 遍历所有参数张量
    for param_tensor in state_dict.values():
        if isinstance(param_tensor, torch.Tensor):
            total_params += param_tensor.numel()  # 总参数量

    print(f"总参数量: {total_params / 1e9} B")
    # print(f"可训练参数量: {trainable_params}")


# 测试
ckpt_path = '/home/xli/skw/MyGPT/ckpt/epoch2_batch_39999'  # 替换为你的.ckpt文件路径
count_parameters(ckpt_path)
