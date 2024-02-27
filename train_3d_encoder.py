import string

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from read_mesh_to_ply import read_mesh_to_ply

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Car3DDataSet(Dataset):
    def __init__(self, directory: str = 'trial_car', length: int = 50, geo_label_file: str = 'geo_result.txt',
                 tex_label_file: str = 'tex_result.txt'):
        self.directory = directory
        self.length = length
        # 读取label
        self.geo_label_file = geo_label_file
        self.tex_label_file = tex_label_file
        geo_labels = []
        tex_labels = []
        with open(geo_label_file, 'r') as f:
            geo_label = []
            for line in f.readlines():
                clean_line = line.strip(string.whitespace)
                end_line = False
                if clean_line.startswith('[['):
                    clean_line = clean_line[2:]
                elif clean_line.endswith(']]'):
                    clean_line = clean_line[:-2]
                    end_line = True
                parts = clean_line.split(' ')
                cur_label = [float(part) if part else None for part in parts]
                for val in cur_label:
                    if val:
                        geo_label.append(val)
                if end_line:
                    geo_labels.append(geo_label[:])
                    geo_label.clear()
                    # print(self.geo_labels)
        with open(tex_label_file, 'r') as f:
            tex_label = []
            for line in f.readlines():
                clean_line = line.strip(string.whitespace)
                end_line = False
                if clean_line.startswith('[['):
                    clean_line = clean_line[2:]
                elif clean_line.endswith(']]'):
                    clean_line = clean_line[:-2]
                    end_line = True
                parts = clean_line.split(' ')
                cur_label = [float(part) if part else None for part in parts]
                for val in cur_label:
                    if val:
                        tex_label.append(val)
                if end_line:
                    tex_labels.append(tex_label[:])
                    tex_label.clear()
                    # print(self.geo_labels)
        self.geo_labels = np.asarray(geo_labels, dtype=np.float32)
        self.tex_labels = np.asarray(tex_labels, dtype=np.float32)
        pointClouds = []
        for i in range(length):
            mesh_name = 'mesh' + str(i)
            pointCloud = read_mesh_to_ply(directory, mesh_name)
            # print(pointCloud)
            vertices = np.asarray(pointCloud.vertices, dtype=np.float32)
            colors = np.asarray(pointCloud.colors, dtype=np.float32)
            colors_subset = colors[:, :3]
            result = np.hstack((vertices, colors_subset))
            pointClouds.append(result)
        self.pointClouds = np.asarray(pointClouds, dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        combined_label = np.concatenate((self.geo_labels[idx], self.tex_labels[idx]), axis=-1)
        point_cloud = self.pointClouds[idx]
        return point_cloud, combined_label



def main():
    dataSet = Car3DDataSet()
    print(dataSet.__len__())
    point_cloud, label = dataSet.__getitem__(0)
    print(point_cloud.shape)
    print(label.shape)


if __name__ == '__main__':
    main()
