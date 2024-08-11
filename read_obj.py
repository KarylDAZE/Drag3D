import numpy as np
import trimesh

def read_obj(filename):
    vertices = []
    uvs = []
    faces_indices = []
    faces = []
    file = open(filename, 'r')
    for line in file:
        if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
            # 解析顶点行
            parts = line.split()
            vertex = [float(part) for part in parts[1:]]
            vertices.append(vertex)
        elif line.startswith('vt'):
            # 解析顶点行
            parts = line.split()
            uv = [float(part) for part in parts[1:]]
            uvs.append(uv)
        elif line.startswith('f'):
            # 解析面行
            parts = line.split()
            face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # 注意obj文件中的索引是从1开始的，这里减去1以匹配numpy数组
            faces.append(face)
            face1 = [[int(split_part) - 1 for split_part in part.split('/')] for part in parts[1:]]  # 注意obj文件中的索引是从1开始的，这里减去1以匹配numpy数组
            faces_indices.append(face1)
    file.close()
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32), np.array(uvs, dtype=np.float64), np.array(faces_indices, dtype=np.int32)

def main():
    read_obj('trial_car/mesh0.obj')


if __name__ == '__main__':
    main()

