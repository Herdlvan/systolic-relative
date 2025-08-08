import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 模型定义（保持原始结构和参数不变）
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


# 设置随机种子确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed()

# 读取训练数据
data = pd.read_csv("/Users/boysdontcry104/Documents/vscode/科研/extracted_values.csv")
print("Data read successfully, head of data:")
print(data.head())  # 打印数据前几行检查数据读取情况

# 提取M、N、K列作为特征矩阵X
X = data[['M', 'N', 'K']].to_numpy()
# 提取Label列作为标签向量y
y = data['Label'].to_numpy().reshape(-1, 1)

# 转换为 PyTorch 张量
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# 统计每个类别的样本数量
unique_labels, label_counts = np.unique(y, return_counts=True)
label_count_dict = {label: count for label, count in zip(unique_labels, label_counts)}

# 按样本数量排序类别
sorted_labels = sorted(label_count_dict.items(), key=lambda x: x[1], reverse=True)
print("\n样本数量最多的10个类别:")
for label, count in sorted_labels[:10]:
    print(f"类别 {label}: {count} 个样本")


# 选择出现频率最高的类别作为验证集来源
def select_validation_indices(X, y, total_val_size=0.2):
    # 按样本数量排序类别
    sorted_labels = sorted(label_count_dict.items(), key=lambda x: x[1], reverse=True)

    total_samples = len(y)
    target_val_size = int(total_samples * total_val_size)
    selected_val_indices = []

    # 从样本最多的类别开始选择
    for label, count in sorted_labels:
        label_indices = np.where(y == label)[0]

        # 如果已选验证集大小达到目标，则停止
        if len(selected_val_indices) >= target_val_size:
            break

        # 计算从当前类别中选取的样本数
        samples_to_take = min(count, target_val_size - len(selected_val_indices))

        # 随机选择样本
        selected_indices = np.random.choice(label_indices, samples_to_take, replace=False)
        selected_val_indices.extend(selected_indices)

    return np.array(selected_val_indices)


# 创建验证集索引（从高频类别中选取）
val_indices = select_validation_indices(X, y, total_val_size=0.08)
print(f"验证集大小: {len(val_indices)} ({len(val_indices) / len(y) * 100:.2f}%)")

# 创建数据集
dataset = TensorDataset(X, y)

# 创建验证集数据加载器
val_dataset = Subset(dataset, val_indices)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 注意：训练集使用全部数据
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数（保持原始参数不变）
num_embeddings = 1025
embedding_dim = 8
hidden_dim = 128
num_classes = 387

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConfigClassifier(num_embeddings, embedding_dim, hidden_dim, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=20
)

# 记录训练过程
train_losses = []
val_accuracies = []
best_accuracy = 0.0
best_epoch = 0

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_dataloader:
        # 移到GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

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

        # 计算梯度范数
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        running_loss += loss.item()

    # 计算平均损失
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # 在验证集上评估
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            M = batch_X[:, 0]
            N = batch_X[:, 1]
            K = batch_X[:, 2]
            outputs = model(M, N, K)
            _, predicted = torch.max(outputs.data, 1)
            batch_y = batch_y.squeeze()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    # 学习率调度
    scheduler.step(accuracy)

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}% *BEST*')
    else:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
              f'Validation Accuracy: {accuracy:.2f}%')

    model.train()

print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")

# 绘制训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy')
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()