import torch
import torch.nn as nn

class ConfigClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_classes):
        super(ConfigClassifier, self).__init__()
        self.embedding_M = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_N = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_K = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = nn.Linear(3 * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, M, N, K):
        embed_M = self.embedding_M(M)
        embed_N = self.embedding_N(N)
        embed_K = self.embedding_K(K)
        concat = torch.cat((embed_M, embed_N, embed_K), dim=1)
        out = self.fc1(concat)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# 实例化模型
model = ConfigClassifier(
    num_embeddings=1000,  # 假设输入 M/N/K 的取值范围是 [0, 999]
    embedding_dim=32,
    hidden_dim=128,
    num_classes=10
)

# 查看 embedding_M 的权重
print("Embedding M 权重：")
print(model.embedding_M.weight.data)

