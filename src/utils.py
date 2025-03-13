"""
author:Bruce Zhao
data:2024
"""
# utils.py
import matplotlib.pyplot as plt
import os


def plot_training_curves(train_losses, train_accuracies, epochs=None, save_path='training_curve.png'):
    """
    绘制训练损失和准确率曲线并保存图像

    参数：
        train_losses (list): 训练损失列表
        train_accuracies (list): 训练准确率列表
        epochs (int, optional): 总训练轮次（用于x轴标签）
        save_path (str): 图像保存路径
    """
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    x = range(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label='Training Loss')
    plt.xlabel('Epoch' if epochs else 'Step')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(x, train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch' if epochs else 'Step')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 自动创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 关闭图像，避免内存泄漏