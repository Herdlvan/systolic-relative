import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import random
import matplotlib.pyplot as plt

# 模型定义
class ConfigClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_classes):
        super(ConfigClassifier, self).__init__()
        # 嵌入表
        self.embedding_M = nn.Embedding(num_embeddings, embedding_dim)
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


# 读取训练数据random_datasets.csv
data = pd.read_csv("D:\迅雷下载\extracted_values.csv")
print("Data read successfully, head of data:")
print(data.head())  # 打印数据前几行检查数据读取情况

# 提取M、N、K列作为特征矩阵X
X = data[['M', 'N', 'K']].to_numpy()
# 提取Label列作为标签向量y
y = data['Label'].to_numpy().reshape(-1, 1)

# 转换为 PyTorch 张量
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# 转换为数据集
dataset = TensorDataset(X, y)

# 确定验证集索引，固定抽取20%
all_indices = list(range(len(dataset)))
random.seed(42)  # 设置固定随机种子，保证每次抽取一致
random.shuffle(all_indices)
val_indices = all_indices[:int(0.2 * len(dataset))]
train_indices = all_indices  # 修正训练集索引

# 使用Subset划分训练集和验证集
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型参数
num_embeddings = 1025
embedding_dim = 8
hidden_dim = 128
num_classes = 387

# 初始化模型
model = ConfigClassifier(num_embeddings, embedding_dim, hidden_dim, num_classes)
model.train()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录每个epoch的平均梯度范数
epoch_gradient_norms = []

# 训练模型
num_epochs =300
for epoch in range(num_epochs):
    epoch_grad_norms = []
    for batch_X, batch_y in train_dataloader:
        # 前向传播
        M = batch_X[:, 0]
        N = batch_X[:, 1]
        K = batch_X[:, 2]
        outputs = model(M, N, K)
        # 计算损失
        batch_y = batch_y.squeeze()
        loss = criterion(outputs, batch_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        # 计算梯度范数字
        optimizer.step()


    # 在验证集上评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            M = batch_X[:, 0]
            N = batch_X[:, 1]
            K = batch_X[:, 2]
            outputs = model(M, N, K)
            _, predicted = torch.max(outputs.data, 1)
            batch_y = batch_y.squeeze()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')
    model.train()

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print("模型已保存为 model.pth")

