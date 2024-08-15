import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
from torch.utils.data import DataLoader
from PointConv import PointConvDensityClsSsg
import dnnlib
from train_3d_encoder import Car3DDataSet
from gui import GET3DWrapper
from mesh import Mesh
import click
import trimesh
import cv2
import numpy as np
from PIL import Image
from read_obj import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def barycentric_coordinates(point, triangle_vertices):
    # 确保输入为numpy数组，并且是(n, 3)形状的二维数组
    point = np.reshape(point, (3, 1))
    triangle_vertices = np.reshape(triangle_vertices, (3, 3))
    triangle_vertices = triangle_vertices.transpose()
    return np.linalg.inv(triangle_vertices) @ point

@click.command()
# Required from StyleGAN2.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--cfg', help='Base configuration', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']),
              default='stylegan2')
# @click.option('--gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), required=True)
# @click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
# @click.option('--gamma', help='R1 regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
# My custom configs
### Configs for inference
@click.option('--resume_pretrain', help='Resume from given network pickle', metavar='[PATH|URL]', type=str)
@click.option('--inference_vis', help='whther we run infernce', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--inference_to_generate_textured_mesh', help='inference to generate textured meshes', metavar='BOOL',
              type=bool, default=False, show_default=False)
@click.option('--inference_save_interpolation', help='inference to generate interpolation results', metavar='BOOL',
              type=bool, default=False, show_default=False)
@click.option('--inference_compute_fid', help='inference to generate interpolation results', metavar='BOOL', type=bool,
              default=False, show_default=False)
@click.option('--inference_generate_geo', help='inference to generate geometry points', metavar='BOOL', type=bool,
              default=False, show_default=False)
### Configs for dataset
@click.option('--data', help='Path to the Training data Images', metavar='[DIR]', type=str, default='./tmp')
@click.option('--camera_path', help='Path to the camera root', metavar='[DIR]', type=str, default='./tmp')
@click.option('--img_res', help='The resolution of image', metavar='INT', type=click.IntRange(min=1), default=1024)
@click.option('--data_camera_mode', help='The type of dataset we are using', type=str, default='shapenet_car',
              show_default=True)
@click.option('--use_shapenet_split', help='whether use the training split or all the data for training',
              metavar='BOOL', type=bool, default=False, show_default=False)
### Configs for 3D generator##########
@click.option('--use_style_mixing', help='whether use style mixing for generation during inference', metavar='BOOL',
              type=bool, default=True, show_default=False)
@click.option('--one_3d_generator', help='whether we detach the gradient for empty object', metavar='BOOL', type=bool,
              default=True, show_default=True)
@click.option('--dmtet_scale', help='Scale for the dimention of dmtet', metavar='FLOAT',
              type=click.FloatRange(min=0, max=10.0), default=1.0, show_default=True)
@click.option('--n_implicit_layer', help='Number of Implicit FC layer for XYZPlaneTex model', metavar='INT',
              type=click.IntRange(min=1), default=1)
@click.option('--feat_channel', help='Feature channel for TORGB layer', metavar='INT', type=click.IntRange(min=0),
              default=16)
@click.option('--mlp_latent_channel', help='mlp_latent_channel for XYZPlaneTex network', metavar='INT',
              type=click.IntRange(min=8), default=32)
@click.option('--deformation_multiplier', help='Multiplier for the predicted deformation', metavar='FLOAT',
              type=click.FloatRange(min=1.0), default=1.0, required=False)
@click.option('--tri_plane_resolution', help='The resolution for tri plane', metavar='INT', type=click.IntRange(min=1),
              default=256)
@click.option('--n_views', help='number of views when training generator', metavar='INT', type=click.IntRange(min=1),
              default=1)
@click.option('--use_tri_plane', help='Whether use tri plane representation', metavar='BOOL', type=bool, default=True,
              show_default=True)
@click.option('--tet_res', help='Resolution for teteahedron', metavar='INT', type=click.IntRange(min=1), default=90)
@click.option('--latent_dim', help='Dimention for latent code', metavar='INT', type=click.IntRange(min=1), default=512)
@click.option('--geometry_type', help='The type of geometry generator', type=str, default='conv3d', show_default=True)
@click.option('--render_type', help='Type of renderer we used', metavar='STR',
              type=click.Choice(['neural_render', 'spherical_gaussian']), default='neural_render', show_default=True)
### Configs for training loss and discriminator#
@click.option('--d_architecture', help='The architecture for discriminator', metavar='STR', type=str, default='skip',
              show_default=True)
@click.option('--use_pl_length', help='whether we apply path length regularization', metavar='BOOL', type=bool,
              default=False, show_default=False)  # We didn't use path lenth regularzation to avoid nan error
@click.option('--gamma_mask', help='R1 regularization weight for mask', metavar='FLOAT', type=click.FloatRange(min=0),
              default=0.0, required=False)
@click.option('--d_reg_interval', help='The internal for R1 regularization', metavar='INT', type=click.IntRange(min=1),
              default=16)
@click.option('--add_camera_cond', help='Whether we add camera as condition for discriminator', metavar='BOOL',
              type=bool, default=True, show_default=True)
## Miscs
# Optional features.
@click.option('--cond', help='Train conditional model', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--freezed', help='Freeze first layers of D', metavar='INT', type=click.IntRange(min=0), default=0,
              show_default=True)
# Misc hyperparameters.
@click.option('--batch-gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1), default=4)
@click.option('--cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768,
              show_default=True)
@click.option('--cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512,
              show_default=True)
@click.option('--glr', help='G learning rate  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0))
@click.option('--dlr', help='D learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.002,
              show_default=True)
@click.option('--map-depth', help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group', help='Minibatch std group size', metavar='INT', type=click.IntRange(min=1), default=4,
              show_default=True)
# Misc settings.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
# @click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid50k', show_default=True)
@click.option('--kimg', help='Total training duration', metavar='KIMG', type=click.IntRange(min=1), default=20000,
              show_default=True)
@click.option('--tick', help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=1,
              show_default=True)  ##
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50,
              show_default=True)  ###
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32', help='Disable mixed-precision', metavar='BOOL', type=bool, default=True,
              show_default=True)  # Let's use fp32 all the case without clamping
@click.option('--nobench', help='Disable cuDNN benchmarking', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=0), default=3,
              show_default=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
# GUI settings.
@click.option('--height', help='GUI H', metavar='INT', type=click.IntRange(min=1), default=1024)
@click.option('--width', help='GUI W', metavar='INT', type=click.IntRange(min=1), default=1024)
@click.option('--radius', help='GUI radius', metavar='FLOAT', type=click.FloatRange(min=0), default=2)
@click.option('--fovy', help='GUI fovy in degree', metavar='FLOAT', type=click.FloatRange(min=0), default=50)



def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    G_kwargs=dnnlib.EasyDict(z_dim=512, w_dim=512,mapping_kwargs=dnnlib.EasyDict())

    G_kwargs.one_3d_generator = opts.one_3d_generator
    G_kwargs.n_implicit_layer = opts.n_implicit_layer
    G_kwargs.deformation_multiplier = opts.deformation_multiplier
    resume_pretrain = opts.resume_pretrain
    G_kwargs.use_style_mixing = opts.use_style_mixing
    G_kwargs.dmtet_scale = opts.dmtet_scale
    G_kwargs.feat_channel = opts.feat_channel
    G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
    G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
    G_kwargs.n_views = opts.n_views

    G_kwargs.render_type = opts.render_type
    G_kwargs.use_tri_plane = opts.use_tri_plane

    G_kwargs.tet_res = opts.tet_res

    G_kwargs.geometry_type = opts.geometry_type

    G_kwargs.data_camera_mode = opts.data_camera_mode
    G_kwargs.channel_base = opts.cbase
    G_kwargs.channel_max = opts.cmax

    G_kwargs.mapping_kwargs.num_layers = 8

    G_kwargs.class_name = 'training.networks_get3d.GeneratorDMTETMesh'
    G_kwargs.fused_modconv_default = 'inference_only'
    get3dWrapper=GET3DWrapper(device=device,G_kwargs=G_kwargs,resume_pretrain="pretrained_model/shapenet_car.pt")


    model = PointConvDensityClsSsg().to(device)  # 创建一个新的模型实例
    model.load_state_dict(torch.load('encoder_model.pth'))  # 从保存的文件中加载模型参数
    batch_size=2
    losses = []
    # 设定损失函数和优化器
    # criterion = nn.MSELoss()  # 假设使用均方误差损失函数，根据实际需求更改
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_loader = DataLoader(dataset=Car3DDataSet(length=2), batch_size=batch_size, shuffle=False)        

    with torch.no_grad():
        mesh_num=1
        # vertices_gen_result=open("vertices_gen_result.txt","w+")
        np.set_printoptions(threshold=np.inf)

        # target_pos_result=open("target_pos_result.txt","w+")
        # target_color_result=open("target_color_result.txt","w+")
        # get_pos_result=open("get_pos_result.txt","w+")
        # get_color_result=open("get_color_result.txt","w+")

        for inputs,targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 将输入数据从(batch_size, 10000, 6)转换为(batch_size, 6, 10000)以适应卷积
            inputs = inputs.permute(0, 2, 1)
            # print(inputs.shape) torch.Size([batch_size, 6, 10000])
            # outputs_combined_labels=targets
            outputs_combined_labels:torch.Tensor = model(inputs[:, :3, :], inputs[:, 3:, :]) #[batch,1024]
            loss=0
            for outputs_combined_label in outputs_combined_labels:
                outputs_combined_label=outputs_combined_label.unsqueeze(0)
                # print(outputs_combined_label.shape)
                mesh_generated=get3dWrapper.generate(ws_geo=outputs_combined_label[:,:512].repeat(1,22,1),ws_tex=outputs_combined_label[:,512:].repeat(1,9,1))
                # vertices_gen_result.write(str(mesh_generated.v.cpu().numpy()))
                get_faces=mesh_generated.f
                gen_vertices_pos:torch.Tensor=mesh_generated.v;
                number_of_points =len(get_faces)
                # print("get mesh v num:"+str(number_of_points))
                get_vertices_pos=gen_vertices_pos[get_faces].mean(dim=1)
                get_vertices_color:torch.Tensor=get3dWrapper.rgb(get_vertices_pos)
                # pointCloud=trimesh.PointCloud(vertices=get_vertices_pos.cpu().detach().numpy(),colors=get_vertices_color.cpu().detach().numpy())
                # pointCloud.show()

                mesh_name = "mesh" + str(mesh_num)
                path='trial_car/'+mesh_name

                vertices, faces, uvs, faces_indices = read_obj(path+".obj")
                mesh_target = trimesh.Trimesh(vertices, faces)

                image_name=path+'_albedo.png'
                texture_image = Image.open(image_name).convert('RGB')

                # 采样点的数量
                number_of_points =len(faces)
                # print("origin mesh v num:"+str(number_of_points))
                # 初始化一个颜色列表
                vertices_colors = []

                points=vertices[faces].mean(axis=1)
                # 遍历所有采样点所在的三角形，并从纹理图像中采样颜色
                for i in range(0, number_of_points):
                    point=points[i]
                    triangle_vertices = vertices[faces_indices[i][:, 0]]
                    u, v, w = barycentric_coordinates(point, triangle_vertices)
                    assert abs(u + v + w - 1.0) < 0.01, u + v + w
                    uv0 = uvs[faces_indices[i][0][1]]
                    uv1 = uvs[faces_indices[i][1][1]]
                    uv2 = uvs[faces_indices[i][2][1]]
                    new_uv = u * uv0 + v * uv1 + w * uv2
                    # new_uv = uv0 # for debug
                    pixel_x = int(new_uv[0] * texture_image.width)
                    pixel_y = int((1 - new_uv[1]) * texture_image.height)
                    pixel_x = max(0, min(texture_image.width - 1, pixel_x))
                    pixel_y = max(0, min(texture_image.height - 1, pixel_y))
                    color = texture_image.getpixel((pixel_x, pixel_y))
                    color=tuple(x/256 for x in color)
                    # 将颜色添加到颜色列表中
                    vertices_colors.append(color)

                target_vertices_pos=torch.tensor(points,device=device, dtype=torch.float32, requires_grad=False)
                target_vertices_color=torch.tensor(vertices_colors,device=device, dtype=torch.float32, requires_grad=False)


                # target_pos_result.write(str(target_vertices_pos.cpu().detach().numpy())+'\n')
                # target_color_result.write(str(target_vertices_color.cpu().detach().numpy())+'\n')
                # get_pos_result.write(str(get_vertices_pos.cpu().detach().numpy())+'\n')
                # get_color_result.write(str(get_vertices_color.cpu().detach().numpy())+'\n')

                # # 计算两个距离矩阵
                # dist_matrix_get_to_target = torch.cdist(get_vertices_pos, target_vertices_pos, p=2)
                # # dist_matrix_target_to_get = torch.cdist(target_vertices_pos, get_vertices_pos, p=2)

                # # 找到最短距离及其对应的索引
                # min_distances_get_to_target, min_indices_get_to_target = torch.min(dist_matrix_get_to_target, dim=1)
                # # min_distances_target_to_get, min_indices_target_to_get = torch.min(dist_matrix_target_to_get, dim=1)
                # # min_distances_get_to_target_result=open("min_distances_get_to_target_result.txt",'w+')
                # # min_distances_get_to_target_result.write(str(min_distances_get_to_target.cpu().detach().numpy())+'\n')
                # # 计算颜色向量的欧氏距离
                # # 对于 get_vertices_pos，每个最近的 target_vertices_pos 及其对应的颜色
                # nearest_target_colors_for_get = target_vertices_color[min_indices_get_to_target]
                # color_distances_get = torch.norm(get_vertices_color - nearest_target_colors_for_get, dim=1)
                # # color_distances_get_result=open("color_distances_get_result.txt",'w+')
                # # color_distances_get_result.write(str(color_distances_get.cpu().detach().numpy())+'\n')

                # # 对于 target_vertices_pos，每个最近的 get_vertices_pos 及其对应的颜色
                # # nearest_get_colors_for_target = get_vertices_color[min_indices_target_to_get]
                # # color_distances_target = torch.norm(target_vertices_color - nearest_get_colors_for_target, dim=1)
                # loss+=min_distances_get_to_target.sum()+color_distances_get.sum()#+min_distances_target_to_get.sum()+color_distances_target.sum()

                # 初始化变量存储最短距离和颜色距离的总和
                sum_min_distances_get_to_target = 0
                sum_color_distances_get = 0

                # 逐点计算 get_vertices_pos 到 target_vertices_pos 的最短距离
                for i in range(get_vertices_pos.shape[0]):
                    with torch.no_grad():
                        # 扩展维度以进行广播
                        distances = torch.norm(target_vertices_pos - get_vertices_pos[i].unsqueeze(0), dim=1)
                        # 找到最短距离及其对应索引
                        _, min_idx = torch.min(distances, dim=0)
                        
                    # 累加最短距离
                    sum_min_distances_get_to_target += torch.norm(target_vertices_pos[min_idx]- get_vertices_pos[i],dim=0)
                    
                    # 计算颜色距离
                    color_distance = torch.norm(target_vertices_color[min_idx]-get_vertices_color[i], dim=0)
                    sum_color_distances_get += color_distance

                # # 初始化变量存储 target 到 get 的最短距离和颜色距离的总和
                # sum_min_distances_target_to_get = 0
                # sum_color_distances_target = 0

                # # 逐点计算 target_vertices_pos 到 get_vertices_pos 的最短距离
                # for i in range(target_vertices_pos.shape[0]):
                #     with torch.no_grad():
                #         # 扩展维度以进行广播
                #         distances = torch.norm(get_vertices_pos - target_vertices_pos[i].unsqueeze(0), dim=1)
                #         # 找到最短距离及其对应索引
                #         _, min_idx = torch.min(distances, dim=0)
                    
                #     # 累加最短距离
                #     sum_min_distances_target_to_get += torch.norm(get_vertices_pos[min_idx]- target_vertices_pos[i],dim=0)
                    
                #     # 计算颜色距离
                #     color_distance = torch.norm(get_vertices_color[min_idx]-target_vertices_color[i], dim=0)
                #     sum_color_distances_target += color_distance

                # 计算总损失
                loss+= sum_min_distances_get_to_target + sum_color_distances_get#+sum_min_distances_target_to_get+sum_color_distances_target
                mesh_num+=1

            # print(f"patch loss: {loss.item():.4f}",end='')


        print(f"Loss: {loss.item():.4f}")
        losses.append(loss.item())

    print(losses)

    with open('result.txt', 'w') as f:
        array_str = np.array2string(outputs_combined_labels.cpu().detach().numpy(), separator=',')
        f.write(array_str)

if __name__ == '__main__':
    main()
