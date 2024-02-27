import numpy as np
import trimesh


def read_data_and_transform_to_numpy(vertex_file: str = 'geometry_vertex_result.txt',
                                     face_file: str = 'geometry_face_result.txt'):
    vertex_data = []
    face_data = []
    with open(vertex_file, 'r') as f:
        line = f.readline()
        line.strip('\n')
        vertex_data = eval(line)
        print(vertex_data)
    with open(face_file, 'r') as f:
        line = f.readline()
        line.strip('\n')
        face_data = eval(line)
        print(face_data)
    return np.array(vertex_data, dtype=np.float64), np.array(face_data, dtype=np.int32)


def main():
    vertex_data, face_data = read_data_and_transform_to_numpy()
    mesh = trimesh.Trimesh(vertices=vertex_data, faces=face_data)
    mesh.show()


if __name__ == '__main__':
    main()
