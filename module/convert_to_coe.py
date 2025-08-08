import numpy as np

def txt_to_coe(input_file, output_file):
    """
    将量化后的权重文件转换为符合指定格式的 COE 文件。
    :param input_file: 输入的 txt 文件路径（包含有符号 8-bit 数据）
    :param output_file: 输出的 coe 文件路径
    """
    # 读取量化后的权重数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            # 跳过注释行或空行
            if line.strip() and not line.startswith("//"):
                # 将每行数据解析为整数列表
                row = list(map(int, line.strip().split()))
                data.extend(row)

    # 转换为 16 进制格式
    hex_data = []
    for value in data:
        # 将有符号整数转换为无符号 8-bit 整数（补码表示）
        if value < 0:
            value = (1 << 8) + value  # 转换为补码
        hex_data.append(f"{value:02X}")  # 转换为 2 位 16 进制字符串

    # 按照每行 8 个值的格式组织数据
    formatted_lines = []
    for i in range(0, len(hex_data), 8):
        line = " ".join(hex_data[i:i+8])  # 每行 8 个值，空格分隔
        if i + 8 < len(hex_data):
            formatted_lines.append(line + ",")  # 中间行以逗号结尾
        else:
            formatted_lines.append(line + ";")  # 最后一行以分号结尾

    # 写入 COE 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write("\n".join(formatted_lines))  # 写入格式化的行
    print(f"已成功将 {input_file} 转换为 {output_file}")

if __name__ == "__main__":
    # 输入的 txt 文件路径
    input_file = "/Users/boysdontcry104/Documents/vscode/科研/module/quantized_weights_signed/quantized_embedding_N.txt"
    # 输出的 coe 文件路径
    output_file = "/Users/boysdontcry104/Documents/vscode/科研/module/quantized_weights_signed/quantized_weights_signed_N.coe"

    # 执行转换
    txt_to_coe(input_file, output_file)