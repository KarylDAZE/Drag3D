import trimesh
from PIL import Image
import numpy as np
import cv2 as cv
from read_obj import *


def barycentric_coordinates(point, triangle_vertices):
    # 确保输入为numpy数组，并且是(n, 3)形状的二维数组
    point = np.reshape(point, (3, 1))
    triangle_vertices = np.reshape(triangle_vertices, (3, 3))
    triangle_vertices = triangle_vertices.transpose()
    return np.linalg.inv(triangle_vertices) @ point


# 示例
# point = np.array([1, 1, 0], dtype=np.float32)
# triangle = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
#
# ans = barycentric_coordinates(point,triangle)

# 加载OBJ模型和其关联的纹理图片

def read_mesh_to_ply(directory_path: str, mesh_path: str) -> trimesh.PointCloud:
    obj_name = directory_path + '/' + mesh_path + '.obj'
    image_name = directory_path + '/' + mesh_path + '_albedo.png'
    image = cv.imread(image_name)
    vertices, faces, uvs, faces_indices = read_obj(obj_name)
    mesh = trimesh.Trimesh(vertices, faces)
    texture_image = Image.open(image_name).convert('RGB')

    # 采样点的数量
    number_of_points = 10000
    points, face_indices = mesh.sample(number_of_points, return_index=True)

    # 初始化一个颜色列表
    vertex_colors = []

    # 遍历所有采样点所在的三角形，并从纹理图像中采样颜色
    for i in range(0, number_of_points):
        face_index = face_indices[i]
        point = points[i]
        triangle_vertices = vertices[faces_indices[face_index][:, 0]]
        u, v, w = barycentric_coordinates(point, triangle_vertices)
        assert abs(u + v + w - 1.0) < 0.01, u + v + w
        uv0 = uvs[faces_indices[face_index][0][1]]
        uv1 = uvs[faces_indices[face_index][1][1]]
        uv2 = uvs[faces_indices[face_index][2][1]]
        new_uv = u * uv0 + v * uv1 + w * uv2
        # new_uv = uv0 # for debug
        pixel_x = int(new_uv[0] * texture_image.width)
        pixel_y = int((1 - new_uv[1]) * texture_image.height)
        pixel_x = max(0, min(texture_image.width - 1, pixel_x))
        pixel_y = max(0, min(texture_image.height - 1, pixel_y))
        # cv.circle(image, (int(uv0[0] * texture_image.width), int(uv0[1] * texture_image.height)), 20, (255, 0, 0), -1, cv.LINE_AA)
        # cv.circle(image, (int(uv1[0] * texture_image.width), int(uv1[1] * texture_image.height)), 20, (0, 255, 0), -1, cv.LINE_AA)
        # cv.circle(image, (int(uv2[0] * texture_image.width), int(uv2[1] * texture_image.height)), 20, (0, 0, 255), -1, cv.LINE_AA)
        # cv.circle(image, (pixel_x, pixel_y), 20, (255, 255, 255), -1, cv.LINE_AA)
        # cv.imshow("Image with Point", image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        color = texture_image.getpixel((pixel_x, pixel_y))
        # 将颜色添加到颜色列表中
        vertex_colors.append(color)

    # 将颜色数组转换为numpy数组并确保形状正确
    vertex_colors = np.array(vertex_colors).reshape((-1, 3))

    # 创建一个新的包含颜色信息的PointCloud对象
    point_cloud = trimesh.PointCloud(vertices=points, colors=vertex_colors)

    # 导出带颜色信息的PLY文件
    output_path = directory_path + '/' + mesh_path + '.ply'
    point_cloud.export(output_path, file_type='ply')
    return point_cloud


def main():
    output = read_mesh_to_ply('trial_car', 'mesh0')
    print(output)

if __name__ == '__main__':
    main()
