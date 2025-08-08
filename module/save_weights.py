import torch

# 加载模型权重
state_dict = torch.load("best_model.pth", map_location='cpu')

# 打开txt文件写入权重信息
with open("model_weights.txt", "w") as f:
    for key, tensor in state_dict.items():
        # 将张量转换为列表字符串
        weight_list = tensor.cpu().numpy().tolist()
        f.write(f"{key}: {weight_list}\n")

print("Model weights saved to model_weights.txt")