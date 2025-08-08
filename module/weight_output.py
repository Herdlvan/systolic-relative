import ast
import numpy as np
import os

def parse_weights(file_path):
    weights = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割层名和权重
            if ':' in line:
                layer_name, weight_str = line.split(':', 1)
                # 将权重字符串解析为 Python 数据结构
                try:
                    weight = ast.literal_eval(weight_str.strip())
                    weights[layer_name.strip()] = np.array(weight)
                except Exception as e:
                    print(f"解析 {layer_name} 的权重时出错: {e}")
    return weights

def print_weights_info(weights):
    for layer_name, weight_array in weights.items():
        print(f"层名: {layer_name}")
        print(f"权重尺寸: {weight_array.shape}")

def save_weights_to_files(weights, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, weight_array in weights.items():
        file_name = f"{layer_name}.txt"
        file_path = os.path.join(output_dir, file_name)
        # 保存权重到文件
        np.savetxt(file_path, weight_array, fmt="%.6f")
        print(f"已保存 {layer_name} 的权重到 {file_path}")

if __name__ == "__main__":
    # 替换为你的 txt 文件路径
    file_path = "model_weights.txt"
    output_dir = "weights_output"  # 输出目录
    weights = parse_weights(file_path)
    print_weights_info(weights)
    save_weights_to_files(weights, output_dir)