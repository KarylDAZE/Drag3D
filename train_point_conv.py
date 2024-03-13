import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader
from PointConv import PointConvDensityClsSsg
from train_3d_encoder import Car3DDataSet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    losses = []
    model = PointConvDensityClsSsg().to(device)
    # 设定损失函数和优化器
    criterion = nn.MSELoss()  # 假设使用均方误差损失函数，根据实际需求更改
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_loader = DataLoader(dataset=Car3DDataSet(), batch_size=5, shuffle=True)
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 将输入数据从(批量大小, 50000, 6)转换为(批量大小, 6, 50000)以适应卷积
            inputs = inputs.permute(0, 2, 1)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs[:, :3, :], inputs[:, 3:, :])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        losses.append(loss.item())

    print(losses)
    torch.save(model.state_dict(), 'encoder_model.pth')


if __name__ == '__main__':
    main()
