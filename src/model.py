"""
author:Bruce Zhao
data:2024
"""
import torch.nn as nn# 神经网络模块（如Linear、ReLU）
# --------------------- 3. 定义全连接网络 ---------------------

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络层：
        self.flatten = nn.Flatten()# 将图像展平为向量（28*28=784维）
        self.fc1 = nn.Linear(28 * 28, 128)# 全连接层1：输入784 → 输出128
        self.relu = nn.ReLU() # 激活函数：引入非线性
        self.fc2 = nn.Linear(128, 64)# 全连接层2：输入128 → 输出64
        self.fc3 = nn.Linear(64, 10)# 输出层：输入64 → 输出10（对应0-9分类）

    def forward(self, x):
        # 定义前向传播路径：
        x = self.flatten(x)# 输入：[batch,1,28,28] → 输出：[batch,784]
        x = self.fc1(x)# → [batch,128]
        x = self.relu(x)# → [batch,128]（ReLU激活）
        x = self.fc2(x)# → [batch,64]
        x = self.relu(x)# → [batch,64]
        x = self.fc3(x)# → [batch,10]（未使用Softmax，因CrossEntropyLoss自带）
        return x