import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from train_3d_encoder import Car3DDataSet
from ResNet import CustomNet
from read_mesh_to_ply import read_mesh_to_ply
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    model = CustomNet(num_output_features=1024).to(device)  # 创建一个新的模型实例
    model.load_state_dict(torch.load('resnet_model.pth'))  # 从保存的文件中加载模型参数

    pointCloud = read_mesh_to_ply('trial_car', 'mesh51')
    vertices = np.asarray(pointCloud.vertices, dtype=np.float32)
    colors = np.asarray(pointCloud.colors, dtype=np.float32)
    colors_subset = colors[:, :3]
    result = np.hstack((vertices, colors_subset))
    print(result.shape)
    input = torch.tensor(result, dtype=torch.float32, device=device)
    input = input.permute(1, 0)
    input_3d = input.unsqueeze(0)
    output = model(input_3d)
    with open('result.txt', 'w') as f:
        f.write(str(output.cpu().detach().numpy()))


if __name__ == '__main__':
    main()
