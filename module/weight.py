import numpy as np
import os

def quantize_to_signed_8bit(weights):
    """
    将权重矩阵量化为有符号 8-bit 数据。
    :param weights: 原始权重矩阵 (numpy array)
    :return: 量化后的有符号 8-bit 数据、缩放因子和偏移量
    """
    # 1. 找到权重的最小值和最大值
    min_val = weights.min()
    max_val = weights.max()

    # 2. 计算缩放因子
    scale = (max_val - min_val) / 255.0  # 缩放因子
    zero_point = -128  # 偏移量调整为有符号范围

    # 3. 量化：将浮点值映射到 [-128, 127]
    quantized = np.round((weights - min_val) / scale + zero_point).astype(np.int8)

    return quantized, scale, min_val

def process_weight_file(input_file, output_dir):
    """
    读取权重文件，量化权重为有符号 8-bit 并保存到新文件。
    :param input_file: 输入权重文件路径
    :param output_dir: 输出目录路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取权重矩阵
    with open(input_file, 'r', encoding='utf-8') as f:
        weights = []
        for line in f:
            # 跳过空行或注释行
            if line.strip() and not line.startswith("//"):
                row = list(map(float, line.strip().split()))
                weights.append(row)
        weights = np.array(weights)

    print(f"权重矩阵尺寸: {weights.shape}")

    # 量化权重
    quantized, scale, min_val = quantize_to_signed_8bit(weights)

    # 保存量化后的权重到文件
    quantized_file = os.path.join(output_dir, "quantized_fc2.weight.txt")
    np.savetxt(quantized_file, quantized, fmt='%d')
    print(f"已保存量化后的权重到 {quantized_file}")

    # 保存量化参数（scale 和 min_val）
    param_file = os.path.join(output_dir, "quantization_params_fc2.weight.txt")
    with open(param_file, 'w', encoding='utf-8') as param_f:
        param_f.write(f"scale: {scale}\n")
        param_f.write(f"min_val: {min_val}\n")
    print(f"已保存量化参数到 {param_file}")

if __name__ == "__main__":
    # 输入权重文件路径
    input_file = "/Users/boysdontcry104/Documents/vscode/科研/module/weights_output/fc2.weight.txt"  # 替换为你的权重文件路径
    output_dir = "/Users/boysdontcry104/Documents/vscode/科研/module/quantized_weights_signed"  # 输出目录

    # 处理权重文件
    process_weight_file(input_file, output_dir)