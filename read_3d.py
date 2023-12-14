import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d.io import load_obj, save_obj
from torch import nn
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib

# set os environment
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

class Car3DDataSet(Dataset):
    def __init__(self, path):
        self.all_3d_cars = []
        for root, dir, files in os.walk(path):
            cnt = 0
            for file in files:
                if file.endswith('.obj'):
                    verts, faces, aux = load_obj(os.path.join(root, file))
                    print("verts.size = ", verts.size())
                    cnt+=1
                    self.all_3d_cars.append((verts, faces, aux))
            print(cnt)

    def __getitem__(self, index):
        return self.all_3d_cars[index], self.all_3d_cars[index][1]

    def __len__(self):
        return len(self.all_3d_cars)

Car3DDataSet('./trial_car')

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyNet().to(device)
print(model)