"""
author:Bruce Zhao
data:2024
"""
from src.data_loader import get_mnist_loaders
from src.model import FCNet
from src.train import train_model
import torch
from src.utils import plot_training_curves

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders()
    model = FCNet().to(device)
    model, losses, acc = train_model(model, train_loader, test_loader, device)

    # 训练完成后调用
    plot_training_curves(
        train_losses=losses,
        train_accuracies=acc,
        epochs=20,  # 明确总epoch数，x轴显示"Epoch"
        save_path='outputs/logs/training_curves.png'  # 自定义保存路径
    )


if __name__ == "__main__":
    main()