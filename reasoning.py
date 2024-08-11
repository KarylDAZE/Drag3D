import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train_3d_encoder import Car3DDataSet
from read_mesh_to_ply import read_mesh_to_ply
import numpy as np
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
    model = PointConvDensityClsSsg().to(device)  # 创建一个新的模型实例
    model.load_state_dict(torch.load('encoder_model.pth'))  # 从保存的文件中加载模型参数

    # pointCloud = read_mesh_to_ply('trial_car', 'mesh0')
    # vertices = np.asarray(pointCloud.vertices, dtype=np.float32)
    # colors = np.asarray(pointCloud.colors, dtype=np.float32)
    # colors_subset = colors[:, :3]
    # result = np.hstack((vertices, colors_subset))
    # print(result.shape)
    # input = torch.tensor(result, dtype=torch.float32, device=device)
    # input = input.permute(1, 0)
    # input_3d = input.unsqueeze(0)
    # print(input_3d.shape)
    data_loader = DataLoader(dataset=Car3DDataSet(length=2), batch_size=2, shuffle=True)
    np.set_printoptions(threshold=np.inf)
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 2, 1)
            # print(inputs)
            output = model(inputs[:, :3, :], inputs[:, 3:, :])
            # with open('targets.txt', 'w') as f:
            #     f.write(str(targets.cpu().detach().numpy()))
    with open('result.txt', 'w') as f:
        array_str = np.array2string(output.cpu().detach().numpy(), separator=',')
        f.write(array_str)


if __name__ == '__main__':
    main()
