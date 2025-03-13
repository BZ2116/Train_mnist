"""
author:Bruce Zhao
data:2024
"""
from torchvision import datasets, transforms# MNIST数据集和预处理
from torch.utils.data import DataLoader# 数据加载器（批处理）

# 定义数据预处理流水线：
# 1. 将图像转换为Tensor（维度从[H, W, C]变为[C, H, W]）
# 2. 归一化：将像素值从[0, 255]缩放到[-1, 1]附近（加速收敛）
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),# 转为Tensor并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))# 使用MNIST的均值和标准差进一步归一化
    ])
    # 下载并加载MNIST数据集：
    # - root: 数据集保存路径
    # - train=True: 加载训练集（6万张）；train=False加载测试集（1万张）
    # - download=True: 如果本地不存在，自动下载
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)# 应用预处理
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 创建数据加载器（DataLoader）：
    # - batch_size每批加载图像数量
    # - shuffle=True: 打乱训练集顺序（防止模型记忆顺序）
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader