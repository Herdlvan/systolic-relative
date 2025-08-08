import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 模型定义
class ConfigClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_classes):
        super(ConfigClassifier, self).__init__()
        # 嵌入表
        self.embedding_M = nn.Embedding(num_embeddings, embedding_dim)  # 1024*8
        self.embedding_N = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_K = nn.Embedding(num_embeddings, embedding_dim)

        # 全连接层
        self.fc1 = nn.Linear(3 * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, M, N, K):
        # 嵌入表映射
        embed_M = self.embedding_M(M)
        embed_N = self.embedding_N(N)
        embed_K = self.embedding_K(K)

        # 拼接特征
        concat = torch.cat((embed_M, embed_N, embed_K), dim=1)

        # 全连接层
        out = self.fc1(concat)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out

# 模型参数
num_embeddings = 1025  # M, N, K 的取值范围是 0 到 1024
embedding_dim = 8  # 嵌入表嵌入维度，可以更改
hidden_dim = 128  # 隐藏层维度
num_classes = 387  # 输出类别数

# 初始化模型
model = ConfigClassifier(num_embeddings, embedding_dim, hidden_dim, num_classes)
# 加载模型权重
model.load_state_dict(torch.load('model.pth'))
# 设置模型为评估模式
model.eval()

# 假设我们有新的推理数据，这里简单模拟一个
new_data = pd.DataFrame({
    'M': [800, 200],
    'N': [189, 400],
    'K': [797, 600]
})
new_X = new_data[['M', 'N', 'K']].to_numpy()
new_X = torch.tensor(new_X, dtype=torch.long)

# 禁用梯度计算
with torch.no_grad():
    # 提取特征
    new_M = new_X[:, 0]
    new_N = new_X[:, 1]
    new_K = new_X[:, 2]
    # 进行推理
    predictions = model(new_M, new_N, new_K)
    # 获取预测的类别
    _, predicted_classes = torch.max(predictions, 1)

print("预测的类别:", predicted_classes)