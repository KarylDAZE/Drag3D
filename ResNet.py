import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train_3d_encoder import Car3DDataSet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# 定义一个接收二维输入的ResNet模型
class CustomNet(nn.Module):
    def __init__(self, num_output_features=1024, num_blocks=None):
        super(CustomNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10000 * 6, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, num_output_features)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    model = CustomNet().to(device)
    # 设定损失函数和优化器
    criterion = nn.MSELoss()  # 假设使用均方误差损失函数，根据实际需求更改
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_loader = DataLoader(dataset=Car3DDataSet(), batch_size=1, shuffle=True)

    # 开始训练
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 将输入数据从(批量大小, 50000, 6)转换为(批量大小, 6, 50000)以适应2D卷积
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'resnet_model.pth')


if __name__ == '__main__':
    main()
