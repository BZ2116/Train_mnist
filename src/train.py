"""
author:Bruce Zhao
data:2024
"""
import torch
from torch import optim, nn# 优化器（如SGD）


def train_model(model, train_loader, test_loader, device, epochs=15, lr=0.1):
    # --------------------- 4. 定义损失函数与优化器 ---------------------
    criterion = nn.CrossEntropyLoss()# 交叉熵损失（多分类任务）
    optimizer = optim.SGD(model.parameters(), lr=lr)# 随机梯度下降，学习率设为0.1（较大以加速收敛）

    # --------------------- 5. 训练模型 ---------------------
    train_losses = []# 记录每轮训练损失
    train_acc = []# 记录每轮训练准确率

    for epoch in range(epochs):  # 训练10个epoch
        model.train()# 设置为训练模式（启用Dropout/BatchNorm等）
        running_loss = 0.0# 累计损失
        correct = 0# 正确预测数
        total = 0# 总样本数
        # 逐批次训练
        for images, labels in train_loader:
            # 将数据转移到GPU（如果可用）
            images, labels = images.to(device), labels.to(device)

            # 前向传播：计算预测值
            outputs = model(images)# 输出形状：[batch,10]
            loss = criterion(outputs, labels)# 计算损失

            # 反向传播：计算梯度
            optimizer.zero_grad()# 清空历史梯度（防止累加）
            loss.backward()# 反向传播计算梯度
            optimizer.step()# 更新模型参数

            # 统计损失和准确率
            running_loss += loss.item()# 累加损失（.item()提取标量值）
            _, predicted = torch.max(outputs.data, 1)# 取预测类别（dim=1找每行最大值索引）
            total += labels.size(0)# 累加当前批次样本数
            correct += (predicted == labels).sum().item()# 统计正确数

        epoch_loss = running_loss / len(train_loader)# 平均损失 = 总损失 / 批次数
        epoch_acc = 100 * correct / total# 准确率 = 正确数 / 总数
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model, train_losses, train_acc