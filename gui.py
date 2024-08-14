import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage.draw import line_aa
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr
import time
import click

from mesh import Mesh

import dnnlib
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

# set os environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.

upfirdn2d._init()
bias_act._init()
filtered_lrelu._init()
conv2d_gradfix.enabled = True  # Improves training speed.
grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

CUBE_V = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
CUBE_E = np.array([[0, 1], [0, 2], [0, 4], [4, 5], [4, 6], [5, 1], [5, 7], [1, 3], [2, 3], [3, 7], [6, 7], [2, 6]])




def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=1000):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(np.radians(self.fovy) / 2)
        aspect = self.W / self.H
        return np.array([[1 / (y * aspect), 0, 0, 0],
                         [0, -1 / y, 0, 0],
                         [0, 0, -(self.far + self.near) / (self.far - self.near),
                          -(2 * self.far * self.near) / (self.far - self.near)],
                         [0, 0, -1, 0]], dtype=np.float32)

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])


class GET3DWrapper:
    def __init__(self, device, G_kwargs, resume_pretrain):

        self.device = device

        common_kwargs = dict(c_dim=0, img_resolution=1024, img_channels=3)
        G_kwargs['device'] = device
        self.G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(
            device)  # subclass of torch.nn.Module

        print('[INFO] resume GET3D from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location='cpu')

        # we only need the ema model
        self.G.load_state_dict(model_state_dict['G_ema'], strict=True)
        del model_state_dict

        # some reference for convenience
        self.num_ws_geo_triplane = self.G.synthesis.generator.tri_plane_synthesis.num_ws_geo
        self.num_ws_tex_triplane = self.G.synthesis.generator.tri_plane_synthesis.num_ws_tex

        self.mesh = None

    @torch.no_grad()
    def rgb(self, pos):
        # pos: [N, 3] torch float tensor
        np.set_printoptions(threshold=np.inf)
        
        # query triplane feature
        tex_feat = self.G.synthesis.generator.get_texture_prediction(self.mesh.tex_feature, pos.unsqueeze(0),
                                                                     self.mesh.ws_tex_last)  # [1, N, C]

        # project to rgb space (to_rgb is 1x1 conv, so we can use it as an MLP)
        rgb = self.G.synthesis.to_rgb(tex_feat.permute(0, 2, 1).contiguous().unsqueeze(-1),
                                      self.mesh.ws_tex_last[:, -1]).squeeze(-1).squeeze(0).t().contiguous()
        rgb = (rgb + 1) / 2
        return rgb

    @torch.no_grad()
    def sample_geo(self, geo_z=None):

        if geo_z is None:
            geo_z = torch.randn([1, self.G.z_dim], device=self.device)
            # for debug
            # geo_z = torch.tensor([[0.753272688, 0.296991819, -1.24820497, 0.787325835, -0.805077827, -0.111713949, -1.62974443, -0.332180953, 0.513218021, -1.60702336, -0.454748595, -0.743706822, -0.492145097, 0.792238331, 0.436733329, 0.38103351, -0.630071104, 0.296452045, 1.89772475, -0.0364924669, 0.0782783329, 0.35657227, 0.0720969513, -0.212371603, 0.906171381, -2.00251603, 0.144659445, 0.881880462, -0.420410305, -0.240807548, -1.77010536, -0.172361955, 0.998687863, -0.369899124, 0.791702032, 1.15648794, -0.29208082, 0.494948417, 0.439457148, -0.333707035, -1.00744998, 0.392038286, -0.5348804, -0.241231173, -0.935283542, -2.01609421, -0.208738327, -0.325110376, -0.408642977, -0.718103468, 0.532744527, -0.446461588, 1.44745874, 0.103139445, -0.341785818, -1.3401103, 0.0313206837, -1.09517884, 0.841431975, -0.121245533, -0.310638368, -0.386945367, 2.01985383, -1.28647256, 0.164970204, -0.129513085, -0.788557589, 0.501663268, 1.48451293, 0.702844679, -0.699007034, 0.293900371, -2.10779524, 1.88116479, -0.969480217, -0.593614042, -0.230343804, 0.491945535, -1.21341383, -0.784106851, -0.266250938, 0.791017175, -0.694055438, 0.402807325, -0.315916389, -0.0279268213, 0.402846396, -0.300660282, -0.188558728, -1.0962491, -0.496932507, -1.54798639, 0.676095903, -0.104174674, 0.276528001, 1.6510607, -0.283255249, -0.448999763, 0.691247642, 0.409478009, -1.35835564, 0.593913198, -1.63339972, 0.142450154, -1.82102418, 1.10099316, 0.244296417, -1.53568745, -1.14467084, -1.60373342, 0.547231734, 0.229446903, -0.384024471, -0.137766585, -0.00535747595, -0.628430903, 0.648441315, -0.224983826, -0.685280085, 0.25190556, 0.929639935, 0.851748109, -0.592133343, 0.575885594, 0.953272462, 1.67548382, 0.571040213, 1.3409065, -0.781355143, 0.464828491, 0.182650343, -1.25280762, 0.378793538, -0.00213895622, -0.318833649, -0.347590894, -0.301067561, -1.86655402, -0.380036473, 1.32175112, 1.44079161, 2.37885261, 1.018049, 0.0352680385, 0.963887155, -0.440470517, 1.3182658, 1.48888409, 0.404103726, -0.191535443, -0.28713724, -0.102475874, 1.28552961, -0.734030902, 2.0983274, 0.668867946, -1.26624763, -0.0436506197, 1.03272605, -1.24680817, 0.811740696, 0.144679025, -0.793309808, -0.298471183, -2.30150533, -0.698901296, 1.9119153, -0.399982125, -0.0437663421, -0.497897446, -0.99044162, 0.0568263121, 0.370549113, -2.00468159, -0.035708461, -1.39991927, -0.773647487, -1.73149836, 0.36512956, -0.291350603, 0.731193006, -0.509116054, -0.663447738, 0.344364017, -0.78793484, 0.20211342, -1.84420717, 0.816509724, -0.42424944, -1.24472654, -0.949420989, -0.347996891, 1.25970054, -0.126141757, -0.979447246, 1.0182265, -0.690194607, 0.0593707822, 0.319132656, 0.268361121, 0.545027971, 0.532950342, -1.16109097, 0.231194749, 0.406146795, -0.437881142, -0.148385525, -0.310033262, -0.120783754, 0.337275475, 3.08132744, -0.412988782, 0.468108416, -0.0951634124, 1.46477783, 1.0672853, -0.698983788, -1.08916152, 1.38075805, -0.845378518, -0.78315264, -1.31151199, 0.377555162, 0.522754788, -1.79078507, 0.0522151254, -0.972731471, -0.0671110675, 0.409369558, -0.460662901, 0.350436568, -0.530073285, -0.591002822, -0.599092364, 1.14860213, 0.69147706, -0.602198482, -0.551573992, 1.70442533, -1.09332299, 0.44047448, 0.215262726, 0.10274604, -0.0996476561, -0.202015564, -1.45247138, 0.22797513, 0.405330926, -1.58915424, -0.333203256, -0.256792337, 1.32750213, 1.55872142, -1.0536685, -0.339072496, -0.891303301, 0.609531879, -1.62904906, 0.243397459, 0.0150706964, 0.928118706, 0.138376459, -0.379651517, 0.245403707, 0.804626524, 0.0898758918, 1.44875264, -0.492684156, 0.293376535, -0.104835749, 0.236348867, -1.22827613, 0.912163258, 0.605282903, -1.58374715, -0.508844852, 0.823317945, 1.02127779, 0.649007022, -0.497948855, -1.1073842, 0.508327782, -1.27743638, -0.521933019, -0.206576452, 0.635091066, -2.36031818, -0.908052683, -1.33847213, -0.249295071, -1.31569588, -1.55560744, -1.25010335, -0.567612231, -1.50538695, -0.478015423, -0.188747808, 1.27458239, 1.12427104, -0.736912131, 2.37125087, 0.0530954078, 2.10840893, -0.620102286, 0.654610753, 0.289151907, 0.295930207, -1.421525, 1.04861975, -0.292822778, -0.888429701, 0.821510911, -1.24767315, -2.80969739, -0.624545753, 0.126416132, 0.357494503, -1.04874992, 0.274019927, -0.825762272, -1.44125426, -0.0428592153, 0.746896863, -0.512284458, -0.125718489, 0.31948337, 1.26653457, -0.830724299, 0.825067401, 0.0176626109, -0.225348517, -0.181749567, 0.404600382, 0.682323813, -0.992933035, 0.761570334, -0.213552728, 1.36356044, 1.50360668, 1.44184971, -1.96018505, 0.406603634, 0.00298209791, 0.866151512, 1.69401455, 1.44958615, 0.936862648, 1.56871414, 0.806346655, -1.51373255, 0.0661646351, -0.00237210025, -0.177940547, -0.00518651819, -0.955150545, -0.329158008, 0.231181249, -0.187004939, 0.337267876, -1.92224932, 1.39618838, 0.643569708, 0.724248946, 2.20512795, 1.6377238, 0.237934917, -1.6335026, 0.0559253283, 0.763756335, 0.763623714, -1.42413962, -0.613972545, -0.308091849, 0.70613265, 0.205167979, -0.360122472, -1.24555421, -1.85179865, 0.421482593, -0.810388327, 0.400969952, 1.97002959, -2.3589735, -1.47887504, 1.05475092, -0.871976495, 0.816008508, 1.00557303, 0.104562581, -0.949092984, 0.146504536, 0.325388551, -0.370379359, -0.461832017, -0.0555030927, 1.25012577, 0.149461895, -0.0460594147, 0.499283552, -0.321211249, -1.42435396, 2.10887146, 0.731175721, 0.0839683712, 1.47635305, -0.990960896, 0.347044438, 0.301298678, -1.3066144, -0.811994612, 0.0497791395, -0.0772253871, 0.37758407, 0.426294833, 1.36589348, -0.148961574, -0.557963312, -0.311530173, 0.271318078, -0.122376218, 2.67683554, -0.414072812, -0.890226781, -1.66619432, -0.215325817, -1.51884186, -1.23421586, 0.230474636, 0.944761634, 0.146130413, 1.4347508, -1.01470017, 1.97903466, -0.395918369, -0.365335464, 0.434745252, 1.0197078, -3.07724714, -0.417604089, -1.54624212, 0.971958518, 0.136043891, 0.0833265036, -1.07920337, -1.04677474, -1.38201964, 1.76570952, 1.50161648, 0.359305739, 0.759262383, -0.661077201, -1.12733257, 0.0419325791, 0.83561182, 0.779848635, 1.3710649, 0.830302536, -1.59874558, -1.50013065, 1.68919599, -3.05788898, -0.401770383, -0.724525511, 2.32738161, 0.217601717, 0.922135532, -0.334319741, 0.473627716, 0.422929972, -1.16089177, -1.32980669, -0.121641159, 2.01644778, -0.415033132, 0.155970722, 0.469291061, 1.89295363, 0.72506249, 0.671732605, 0.695818901, -0.691303194, 0.241862342, 0.404909343, 0.0972672179, -0.436567605, -0.773598194, -0.325885117, -2.12995434, 0.237614661, -1.14276254, 0.697127402, 0.255344808, 0.0653533265, -0.731938541, -0.189940691, 0.18776916, -1.09955335, -1.43824184, -1.03059328, -1.63221061, -0.311439395, -0.985448897, -1.65505588, -1.15317369, -0.733492672, -0.130378023, 0.336827576, 0.833500743, 0.101183645, 0.360810339, -0.411266714, -0.697299063]], dtype=torch.float32,device=self.device)

        ws_geo = self.G.mapping_geo(geo_z, None, truncation_psi=0.7, truncation_cutoff=None,
                                    update_emas=False)  # [1, 22, 512]
        np.set_printoptions(threshold=np.inf)
        geo_output = open('geo_result.txt', 'a+')
        geo_output.write(str(np.asarray(ws_geo[:,0].cpu())) + '\n')
        # print(str(np.asarray(ws_geo.cpu()),str(np.asarray(geo_z.cpu()))))
        return ws_geo

    @torch.no_grad()
    def sample_tex(self, tex_z=None):

        if tex_z is None:
            tex_z = torch.randn([1, self.G.z_dim], device=self.device)
            # print(tex_z)

        ws_tex = self.G.mapping(tex_z, None, truncation_psi=0.7, truncation_cutoff=None,
                                update_emas=False)  # [1, 9, 512]
        tex_output = open('tex_result.txt', 'a+')
        tex_output.write(str(np.asarray(ws_tex[:,0].cpu())) + '\n')
        return ws_tex

    def generate(self, ws_geo=None, ws_tex=None, geo_z=None, tex_z=None):

        if ws_geo is None:
            ws_geo = self.sample_geo(geo_z)

        if ws_tex is None:
            ws_tex = self.sample_tex(tex_z)

        sdf_feature, tex_feature = self.G.synthesis.generator.get_feature(
            ws_tex[:, :self.num_ws_tex_triplane],  # 7
            ws_geo[:, :self.num_ws_geo_triplane]  # 20
        )  # [1, 96, 256, 256] x 2, triplane features

        ws_tex_last = ws_tex[:, self.num_ws_tex_triplane:].contiguous()  # [1, 2, 512]
        ws_geo_last = ws_geo[:, self.num_ws_geo_triplane:].contiguous()  # [1, 2, 512]

        # geometry
        v, f, sdf, deformation, v_deformed, sdf_reg_loss = self.G.synthesis.get_geometry_prediction(ws_geo_last,
                                                                                                    sdf_feature)
        # v_on_cpu = v[0].cpu().detach().numpy()
        # v_list_data = v_on_cpu.tolist()
        # output_geometry_vertex.write(str(v_list_data) + '\n')
        # f_on_cpu = f[0].cpu().detach().numpy()
        # f_list_data = f_on_cpu.tolist()
        # output_geometry_faces.write(str(f_list_data) + '\n')

        np.set_printoptions(threshold=np.inf)
        # print("v:"+str(v[0].cpu().detach().numpy().shape)) #[9972,3]
        # print("f:"+str(f[0].cpu().detach().numpy().shape)) #[20068,3]

        # build mesh object
        self.mesh = Mesh(v=v[0].float().contiguous(), f=f[0].int().contiguous(), device=self.device)
        self.mesh.auto_normal()

        # bind features to mesh for convenience
        self.mesh.sdf_feature = sdf_feature
        self.mesh.ws_geo = ws_geo
        self.mesh.ws_geo_last = ws_geo_last
        self.mesh.tex_feature = tex_feature
        self.mesh.ws_tex = ws_tex
        self.mesh.ws_tex_last = ws_tex_last
        # sdf_feature_output.write(str(np.asarray(sdf_feature)) + '\n')
        # tex_feature_output.write(str(np.asarray(tex_feature)) + '\n')
        # print(dir(self.mesh))
        return self.mesh


def make_offsets(r, device):
    p = torch.arange(-r, r + 1, device=device)
    px, py, pz = torch.meshgrid(p, p, p, indexing='ij')
    offsets = torch.stack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)], dim=-1)  # [B = (2 * r1 + 1) ** 3, 3]
    return offsets


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.light_dir = np.array([0, 0])
        self.mode = 'lambertian'
        self.ambient_ratio = 0.5
        self.save_path = 'mesh0.obj'
        # file_path
        self.load_file_path = None

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_overlay = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.buffer_rast = None  # for 2D to 3D projection

        self.need_update = True  # update buffer_image
        self.need_update_overlay = True  # update buffer_overlay

        self.glctx = dr.RasterizeCudaContext()  # dr.RasterizeGLContext()

        # load model
        self.device = torch.device('cuda')
        self.model = GET3DWrapper(self.device, G_kwargs=opt.G_kwargs, resume_pretrain=opt.resume_pretrain)

        # current generated mesh and latent codes
        self.mesh = None

        # drag stuff
        self.mouse_loc = np.array([0, 0])
        self.point_idx = 0

        self.point_3d = [0, 0, 0]
        self.points_3d = []
        self.points_mask = []
        self.points_3d_delta = []

        self.bbox_points = [[-1, -1, -1], [1, 1, 1]]
        self.enable_bbox_loss = False
        # TODO: a suitable batch size and weight
        self.bbox_batch = 4096
        self.bbox_loss_weight = 1  # 20 in paper...

        # training stuff
        self.training = False
        self.optimizer = None
        self.ws_geo_param = None
        self.ws_geo_nonparam = None
        self.lr = 0.002
        self.r1 = 3  # 3
        self.r2 = 9  # 12

        self.offsets1 = make_offsets(self.r1, self.device)
        self.offsets2 = make_offsets(self.r2, self.device)

        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        dpg.create_context()
        self.register_dpg()
        self.test_step()

        self.num_gen=0

    def __del__(self):
        dpg.destroy_context()

    def prepare_train(self):
        assert self.mesh is not None, 'must generate a mesh before training'
        assert len(self.points_mask) > 0 and np.any(
            self.points_mask), 'must mark at least a drag point pair before training'

        # TODO: optimize how many layers is the best? need to be verified...
        # self.ws_geo_param = torch.nn.Parameter(self.mesh.ws_geo.clone()) # [1, l, 512]
        # self.optimizer = torch.optim.AdamW([self.ws_geo_param], lr=self.lr)

        # l_start = 4 # skip optimizing early feature layers as suggested in paper
        l_start = 4  # try to optimize an early feature layer to generate a better result for large scale drag
        self.ws_geo_nonparam = self.mesh.ws_geo[:, :l_start].clone()  # [1, l, 512]
        self.ws_geo_param = torch.nn.Parameter(self.mesh.ws_geo[:, l_start:].clone())  # [1, 22-l, 512]
        self.optimizer = torch.optim.Adam([self.ws_geo_param], lr=self.lr)

        self.step = 0

        # keep a copy of the original sdf feature for bbox loss
        if self.enable_bbox_loss:
            self.original_sdf_feature = self.mesh.sdf_feature.detach().clone()

    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1

            ### 3D patch feature loss
            # loss --> triplanes (sdf_feature) --> ws_geo
            # ws_geo = self.ws_geo_param
            ws_geo = torch.cat([self.ws_geo_nonparam, self.ws_geo_param], dim=1)
            ws_tex = self.mesh.ws_tex
            sdf_feature, tex_feature = self.model.G.synthesis.generator.get_feature(
                ws_tex[:, :self.model.num_ws_tex_triplane],  # 7
                ws_geo[:, :self.model.num_ws_geo_triplane]  # 20
            )  # [1, 96, 256, 256] x 2, triplane features

            # get drag point pairs (no need to have grad)
            with torch.no_grad():
                mask_points = torch.tensor(self.points_mask, dtype=torch.bool, device=self.device)
                source_points = torch.tensor(self.points_3d, dtype=torch.float32, device=self.device)[
                    mask_points]  # [N, 3]
                target_points = source_points + \
                                torch.tensor(self.points_3d_delta, dtype=torch.float32, device=self.device)[mask_points]
                directions = safe_normalize(target_points - source_points)

                resolution = sdf_feature.shape[-1]  # 256
                step_size = 0.1 / resolution  # critical! should be small enough to make point tracking possible

                # expand source to a patch based on radius
                patched_points = source_points.unsqueeze(0) + step_size * self.offsets1.unsqueeze(1)  # [B, N, 3]
                B, N = patched_points.shape[:2]

                # shift points
                shifted_points = patched_points + step_size * directions  # [B, N, 3]

            # query feat and calc loss
            patched_feat = self.model.G.synthesis.generator.get_sdf_def_prediction(sdf_feature,
                                                                                   patched_points.reshape(1, -1, 3),
                                                                                   return_feats=True).reshape(B, N,
                                                                                                              -1)  # [B, N, C]
            shifted_feat = self.model.G.synthesis.generator.get_sdf_def_prediction(sdf_feature,
                                                                                   shifted_points.reshape(1, -1, 3),
                                                                                   return_feats=True).reshape(B, N,
                                                                                                              -1)  # [B, N, C]

            loss = F.l1_loss(shifted_feat, patched_feat.detach())

            # randomized bounding box mask loss
            if self.enable_bbox_loss:
                # sample random points in side mesh's aabb
                vmin, vmax = self.mesh.aabb()
                rand_points = torch.rand([self.bbox_batch, 3], dtype=torch.float32, device=self.device) * (
                            vmax - vmin) + vmin  # [B, 3]
                # remove those falling into our bbox
                in_bbox_mask = (rand_points > torch.tensor(self.bbox_points[0], dtype=torch.float32,
                                                           device=rand_points.device)).all(dim=1) & \
                               (rand_points < torch.tensor(self.bbox_points[1], dtype=torch.float32,
                                                           device=rand_points.device)).all(dim=1)  # [B]
                rand_points = rand_points[~in_bbox_mask]
                # keep out-of-bbox points feature fixed
                if rand_points.shape[0] > 0:
                    with torch.no_grad():
                        original_feat = self.model.G.synthesis.generator.get_sdf_def_prediction(
                            self.original_sdf_feature, rand_points.reshape(1, -1, 3), return_feats=True)
                    current_feat = self.model.G.synthesis.generator.get_sdf_def_prediction(sdf_feature,
                                                                                           rand_points.reshape(1, -1,
                                                                                                               3),
                                                                                           return_feats=True)
                    bbox_loss = F.l1_loss(current_feat, original_feat)
                    loss = loss + self.bbox_loss_weight * bbox_loss

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            ### point tracking (update points_3d)
            with torch.no_grad():
                source_feat = patched_feat[(B - 1) // 2]  # [N, C]

                # expand source to a patch based on a larger radius
                patched_points = source_points.unsqueeze(0) + step_size * self.offsets1.unsqueeze(1)  # [B, N, 3]
                B, N = patched_points.shape[:2]

                # calculate updated sdf_feature
                # ws_geo = self.ws_geo_param
                ws_geo = torch.cat([self.ws_geo_nonparam, self.ws_geo_param], dim=1)
                ws_tex = self.mesh.ws_tex
                new_sdf_feature, new_tex_feature = self.model.G.synthesis.generator.get_feature(
                    ws_tex[:, :self.model.num_ws_tex_triplane],  # 7
                    ws_geo[:, :self.model.num_ws_geo_triplane]  # 20
                )  # [1, 96, 256, 256] x 2, triplane features

                new_patched_feat = self.model.G.synthesis.generator.get_sdf_def_prediction(new_sdf_feature,
                                                                                           patched_points.reshape(1, -1,
                                                                                                                  3),
                                                                                           return_feats=True).reshape(B,
                                                                                                                      N,
                                                                                                                      -1)  # [B, N, C]

                # nearest neighbor
                dist = torch.mean((new_patched_feat - source_feat) ** 2, dim=-1)  # [B, N]
                # dist[(B - 1) // 2] = 1e8 # forbid always staying in the same point...
                indices = torch.argmin(dist, dim=0)  # [N]
                # print(indices)

                # update points_3d and delta
                new_source_points = torch.gather(patched_points, dim=0,
                                                 index=indices.view(1, -1, 1).repeat(1, 1, 3)).squeeze(1)  # [N, 3]
                new_points_delta = target_points - new_source_points  # [N, 3]

                # need to add back those deleted points... this should be improved...
                new_source_points_with_deleted = np.array(self.points_3d)
                new_source_points_with_deleted[np.array(self.points_mask)] = new_source_points.detach().cpu().numpy()
                new_source_points_delta_with_deleted = np.array(self.points_3d_delta)
                new_source_points_delta_with_deleted[
                    np.array(self.points_mask)] = new_points_delta.detach().cpu().numpy()

                self.points_3d = new_source_points_with_deleted.tolist()
                self.points_3d_delta = new_source_points_delta_with_deleted.tolist()

        # update mesh for rendering
        with torch.no_grad():
            # update geometry
            v, f, sdf, deformation, v_deformed, sdf_reg_loss = self.model.G.synthesis.get_geometry_prediction(
                self.mesh.ws_geo_last, new_sdf_feature)
            # sdf_output.write(str(np.asarray(sdf.cpu()).tolist()) + '\n')
            # build mesh object
            mesh = Mesh(v=v[0].float().contiguous(), f=f[0].int().contiguous(), device=self.device)
            mesh.auto_normal()

            ws_tex_last = ws_tex[:, self.model.num_ws_tex_triplane:].contiguous()  # [1, 2, 512]
            ws_geo_last = ws_geo[:, self.model.num_ws_geo_triplane:].contiguous()  # [1, 2, 512]

            # bind features to mesh for convenience
            mesh.sdf_feature = new_sdf_feature
            mesh.ws_geo = ws_geo
            mesh.ws_geo_last = ws_geo_last
            mesh.tex_feature = new_tex_feature
            mesh.ws_tex = ws_tex
            mesh.ws_tex_last = ws_tex_last

            self.mesh = self.model.mesh = mesh

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        # decide if should stop training (exceed max step or all drag pairs are very close)
        if np.all(np.linalg.norm(np.array(self.points_3d_delta), axis=-1) < 1e-2):
            self.training = False
            dpg.configure_item("_button_train", label="start")

        self.need_update = True
        self.need_update_overlay = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}')

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):

        # ignore if no need to update
        if not self.need_update and not self.need_update_overlay:
            return

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:

            if self.mesh is not None:

                # do MVP for vertices
                mv = torch.from_numpy(self.cam.view).cuda()  # [4, 4]
                proj = torch.from_numpy(self.cam.perspective).cuda()  # [4, 4]
                mvp = proj @ mv

                v_clip = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0),
                                      torch.transpose(mvp, 0, 1)).float().unsqueeze(0)  # [1, N, 4]

                rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (self.H, self.W))

                if self.mode == 'depth':
                    depth = rast[0, :, :, [2]]  # [H, W, 1]
                    buffer_image = depth.detach().cpu().numpy().repeat(3, -1)  # [H, W, 3]
                else:
                    pos, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f)  # [1, H, W, 3]

                    pos = pos.view(-1, 3)
                    mask = (rast[..., 3] > 0).view(-1)
                    albedo = torch.zeros_like(pos, dtype=torch.float32)
                    if mask.any():
                        albedo[mask] = self.model.rgb(pos[mask])
                    albedo = albedo.view(1, self.H, self.W, 3)
                    alpha = (rast[..., [3]] > 0).float()

                    albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).clamp(0, 1)  # [1, H, W, 3]
                    alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).clamp(0, 1)  # [1, H, W, 1]

                    if self.mode == 'albedo':
                        buffer_image = albedo[0]
                    else:
                        normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                        normal = safe_normalize(normal)
                        if self.mode == 'normal':
                            buffer_image = (normal[0] + 1) / 2
                        elif self.mode == 'lambertian':
                            light_d = np.deg2rad(self.light_dir)
                            light_d = np.array([
                                np.sin(light_d[0]) * np.sin(light_d[1]),
                                np.cos(light_d[0]),
                                np.sin(light_d[0]) * np.cos(light_d[1]),
                            ], dtype=np.float32)
                            light_d = torch.from_numpy(light_d).to(albedo.device)
                            lambertian = self.ambient_ratio + (1 - self.ambient_ratio) * (
                                        normal @ light_d).float().clamp(min=0)
                            buffer_image = (albedo * lambertian.unsqueeze(-1))[0]

                    # mix background
                    buffer_image = buffer_image * alpha + self.bg_color.to(buffer_image.device) * (1 - alpha)

                self.buffer_image = buffer_image.detach().cpu().numpy()
                self.buffer_rast = rast
            self.need_update = False

        # should update overlay
        if self.need_update_overlay:
            buffer_overlay = np.zeros_like(self.buffer_overlay)

            mv = self.cam.view  # [4, 4]
            proj = self.cam.perspective  # [4, 4]
            mvp = proj @ mv

            mask = np.array(self.points_mask).astype(bool)

            if mask.any():
                # do mvp transform for keypoints
                source_points = np.array(self.points_3d)[mask]
                target_points = source_points + np.array(self.points_3d_delta)[mask]
                points_indices = np.arange(len(self.points_3d))[mask]

                source_points_clip = np.matmul(np.pad(source_points, ((0, 0), (0, 1)), constant_values=1.0),
                                               mvp.T)  # [N, 4]
                target_points_clip = np.matmul(np.pad(target_points, ((0, 0), (0, 1)), constant_values=1.0),
                                               mvp.T)  # [N, 4]
                source_points_clip[:, :3] /= source_points_clip[:, 3:]  # perspective division
                target_points_clip[:, :3] /= target_points_clip[:, 3:]  # perspective division

                source_points_2d = (((source_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(
                    np.int32)
                target_points_2d = (((target_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(
                    np.int32)

                radius = int((self.H + self.W) / 2 * 0.005)
                for i in range(len(source_points_clip)):
                    point_idx = points_indices[i]
                    # draw source point
                    if source_points_2d[i, 0] >= radius and source_points_2d[i, 0] < self.W - radius and \
                            source_points_2d[i, 1] >= radius and source_points_2d[i, 1] < self.H - radius:
                        buffer_overlay[source_points_2d[i, 1] - radius:source_points_2d[i, 1] + radius,
                        source_points_2d[i, 0] - radius:source_points_2d[i, 0] + radius] += np.array(
                            [1, 0, 0]) if not point_idx == self.point_idx else np.array([1, 0.87, 0])
                        # draw target point
                        if target_points_2d[i, 0] >= radius and target_points_2d[i, 0] < self.W - radius and \
                                target_points_2d[i, 1] >= radius and target_points_2d[i, 1] < self.H - radius:
                            buffer_overlay[target_points_2d[i, 1] - radius:target_points_2d[i, 1] + radius,
                            target_points_2d[i, 0] - radius:target_points_2d[i, 0] + radius] += np.array(
                                [0, 0, 1]) if not point_idx == self.point_idx else np.array([0.5, 0.5, 1])
                        # draw line
                        rr, cc, val = line_aa(source_points_2d[i, 1], source_points_2d[i, 0], target_points_2d[i, 1],
                                              target_points_2d[i, 0])
                        in_canvas_mask = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
                        buffer_overlay[rr[in_canvas_mask], cc[in_canvas_mask]] += val[in_canvas_mask, None] * np.array(
                            [0, 1, 0]) if not point_idx == self.point_idx else np.array([0.5, 1, 0])

            # draw bbox
            if self.enable_bbox_loss:
                bbox_points = np.take_along_axis(np.array(self.bbox_points), CUBE_V, axis=0)  # [8, 3]
                bbox_points_clip = np.matmul(np.pad(bbox_points, ((0, 0), (0, 1)), constant_values=1.0),
                                             mvp.T)  # [N, 4]
                bbox_points_clip[:, :3] /= bbox_points_clip[:, 3:]  # perspective division
                bbox_points_2d = (((bbox_points_clip[:, :2] + 1) / 2) * np.array([self.H, self.W])).round().astype(
                    np.int32)

                for e in CUBE_E:
                    rr, cc, val = line_aa(bbox_points_2d[e[0], 1], bbox_points_2d[e[0], 0], bbox_points_2d[e[1], 1],
                                          bbox_points_2d[e[1], 0])
                    in_canvas_mask = (rr >= 0) & (rr < self.H) & (cc >= 0) & (cc < self.W)
                    buffer_overlay[rr[in_canvas_mask], cc[in_canvas_mask]] += val[in_canvas_mask, None] * np.array(
                        [0.5, 0.5, 0.5])  # grey

            self.buffer_overlay = buffer_overlay
            self.need_update_overlay = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000 / t)} FPS)')

        # mix image and overlay
        # buffer = np.clip(self.buffer_image + self.buffer_overlay, 0, 1) # mix mode, sometimes unclear
        overlay_mask = self.buffer_overlay.sum(axis=-1, keepdims=True) == 0
        buffer = self.buffer_image * overlay_mask + self.buffer_overlay  # replace mode

        dpg.set_value("_texture", buffer)

    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.buffer_image, format=dpg.mvFormat_Float_rgb, tag="_texture")

        def callback_file_select(sender, app_data, user_data):
            _t = time.time()
            dpg.set_value("_log_get_mesh", f'loading...')
            print(app_data)
            self.load_file_path = app_data['file_path_name']
            print("file_path = ", self.load_file_path)
            dpg.set_value("_log_get_mesh", f'mesh loaded in {time.time() - _t:.4f}s')

        with dpg.file_dialog(
                directory_selector=False, show=False, callback=callback_file_select, tag="file_dialog_id",
                cancel_callback=None, width=700, height=400):
            dpg.add_file_extension(".obj")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=600, height=350):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # rendering stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # mesh stuff
            with dpg.collapsing_header(label="Generate", default_open=True):

                # generate a new mesh
                with dpg.group(horizontal=True):
                    dpg.add_text("GET Mesh: ")

                    def callback_get_mesh(sender, app_data, user_data):
                        _t = time.time()
                        dpg.set_value("_log_get_mesh", f'generating...')

                        if self.mesh is None or user_data == 0:
                            self.mesh = self.model.generate()
                        elif user_data == 1:
                            self.mesh = self.model.generate(ws_tex=self.mesh.ws_tex)
                        else:
                            self.mesh = self.model.generate(ws_geo=self.mesh.ws_geo)

                        self.need_update = True
                        self.need_update_overlay = True
                        self.num_gen+=1
                        torch.cuda.synchronize()
                        dpg.set_value("_log_get_mesh", f'generated in {time.time() - _t:.4f}s')

                    def callback_load_mesh(sender, app_data, user_data):
                        dpg.show_item("file_dialog_id")

                    # resample geo & tex
                    dpg.add_button(label="get", tag="_button_get_mesh", callback=callback_get_mesh, user_data=0)
                    dpg.bind_item_theme("_button_get_mesh", theme_button)
                    # keep tex, resample geo
                    dpg.add_button(label="geo", tag="_button_get_mesh_tex", callback=callback_get_mesh, user_data=1)
                    dpg.bind_item_theme("_button_get_mesh_tex", theme_button)
                    # keep geo, resample tex
                    dpg.add_button(label="tex", tag="_button_get_mesh_geo", callback=callback_get_mesh, user_data=2)
                    dpg.bind_item_theme("_button_get_mesh_geo", theme_button)
                    # keep load, load a user-provided 3D model(GAN Inversion)
                    dpg.add_button(label="load", tag="_button_load_mesh", callback=callback_load_mesh, user_data=3)
                    dpg.bind_item_theme("_button_load_mesh", theme_button)

                    dpg.add_text('', tag="_log_get_mesh")

                # save current mesh
                with dpg.group(horizontal=True):
                    dpg.add_text("Save Mesh: ")

                    def callback_set_save_path(sender, app_data):
                        self.save_path = app_data

                    def callback_save_mesh(sender, app_data):
                        self.save_path = 'mesh'+str(self.num_gen)+'.obj'
                        os.makedirs(self.opt.outdir, exist_ok=True)
                        path = os.path.join(self.opt.outdir, self.save_path)
                        print(f'[INFO] save mesh to {path}...')

                        # hardcoded texture resolution
                        h = w = 2048

                        # unwrap uv
                        self.mesh.auto_uv()

                        # rgb query
                        uv = self.mesh.vt * 2.0 - 1.0  # uvs to range [-1, 1]
                        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])),
                                       dim=-1)  # [N, 4]

                        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), self.mesh.ft, (h, w))  # [1, h, w, 4]
                        xyzs, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f)  # [1, h, w, 3]

                        # masked query 
                        xyzs = xyzs.view(-1, 3)
                        mask = (rast[..., 3] > 0).view(-1)

                        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)

                        if mask.any():
                            # print(xyzs.shape) #[h*w=4194304,3]
                            xyzs = xyzs[mask]  # [M, 3]
                            # batched inference to avoid OOM
                            all_albedo = []
                            head = 0
                            # print(xyzs.shape) #[2682636,3]
                            while head < xyzs.shape[0]:
                                tail = min(head + 640000, xyzs.shape[0])
                                all_albedo.append(self.model.rgb(xyzs[head:tail]))
                                head += 640000
                            albedo[mask] = torch.cat(all_albedo, dim=0)

                        albedo = albedo.view(h, w, -1)
                        mask = mask.view(h, w)

                        albedo = albedo.detach().cpu().numpy()
                        mask = mask.detach().cpu().numpy()

                        # dilate texture 
                        from sklearn.neighbors import NearestNeighbors
                        from scipy.ndimage import binary_dilation, binary_erosion

                        inpaint_region = binary_dilation(mask, iterations=32)  # pad width
                        inpaint_region[mask] = 0

                        search_region = mask.copy()
                        not_search_region = binary_erosion(search_region, iterations=3)
                        search_region[not_search_region] = 0

                        search_coords = np.stack(np.nonzero(search_region), axis=-1)
                        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

                        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
                        _, indices = knn.kneighbors(inpaint_coords)

                        albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

                        self.mesh.albedo = torch.from_numpy(albedo).to(self.device)
                        self.mesh.write(path)

                        print(f'[INFO] saved mesh!')

                    dpg.add_button(label="save", tag="_button_save_mesh", callback=callback_save_mesh)
                    dpg.bind_item_theme("_button_save_mesh", theme_button)
                    dpg.add_input_text(label="", default_value=self.save_path, callback=callback_set_save_path)

            # drag stuff
            with dpg.collapsing_header(label="Drag", default_open=True):

                # keypoints list
                def callback_update_keypoint_delta(sender, app_data, user_data):
                    self.points_3d_delta[user_data] = app_data[:3]
                    # update rendering
                    self.need_update_overlay = True

                def callback_delete_keypoint(sender, app_data, user_data):
                    # update states (not really delete since we rely on id... just mark it as deleted)
                    self.points_mask[user_data] = False
                    # update UI (delete group by tag)
                    dpg.delete_item(f"_group_keypoint_{user_data}")
                    # update rendering
                    self.need_update_overlay = True

                def callback_add_keypoint(sender, app_data):
                    # update states
                    self.points_3d.append(self.point_3d)
                    self.points_mask.append(True)
                    self.points_3d_delta.append([0, 0, 0])
                    # update UI
                    _id = len(self.points_3d) - 1
                    dpg.add_group(parent="_group_keypoints", tag=f"_group_keypoint_{_id}", horizontal=True)
                    dpg.add_text(parent=f"_group_keypoint_{_id}",
                                 default_value=f"{', '.join([f'{x:.3f}' for x in self.points_3d[_id]])} +")
                    dpg.add_input_floatx(parent=f"_group_keypoint_{_id}", tag=f"_point_delta_{_id}", size=3, width=200,
                                         format="%.3f", on_enter=True, default_value=self.points_3d_delta[_id],
                                         callback=callback_update_keypoint_delta, user_data=_id)
                    dpg.add_button(parent=f"_group_keypoint_{_id}", tag=f"_button_del_keypoint_{_id}", label="del",
                                   callback=callback_delete_keypoint, user_data=_id)
                    dpg.bind_item_theme(f"_button_del_keypoint_{_id}", theme_button)
                    # update rendering
                    self.need_update_overlay = True

                def callback_update_new_keypoint(sender, app_data):
                    self.point_3d = app_data[:3]

                with dpg.group(horizontal=True):
                    dpg.add_text("Keypoint: ")
                    dpg.add_input_floatx(default_value=self.point_3d, size=3, width=200, format="%.3f", on_enter=False,
                                         callback=callback_update_new_keypoint)
                    dpg.add_button(label="add", tag="_button_add_keypoint", callback=callback_add_keypoint)
                    dpg.bind_item_theme("_button_add_keypoint", theme_button)

                # empty group as a handler
                dpg.add_separator()
                dpg.add_group(tag="_group_keypoints")
                dpg.add_separator()

                # mask area
                def callback_update_bbox_point(sender, app_data, user_data):
                    self.bbox_points[user_data[0]][user_data[1]] = app_data
                    # make sure left-bottom is smaller than right-top
                    if user_data[0] == 0:
                        self.bbox_points[1][user_data[1]] = max(app_data, self.bbox_points[1][user_data[1]])
                        dpg.configure_item(f"_bbox_point_1_{user_data[1]}",
                                           default_value=self.bbox_points[1][user_data[1]])
                    else:
                        self.bbox_points[0][user_data[1]] = min(app_data, self.bbox_points[0][user_data[1]])
                        dpg.configure_item(f"_bbox_point_0_{user_data[1]}",
                                           default_value=self.bbox_points[0][user_data[1]])
                    self.need_update_overlay = True

                def callback_toggle_mask_loss(sender, app_data):
                    self.enable_bbox_loss = not self.enable_bbox_loss
                    self.need_update_overlay = True

                # train stuff
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.training = True
                            self.prepare_train()
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                    dpg.bind_item_theme("_button_train", theme_button)

                    def callback_set_lr(sender, app_data):
                        self.lr = app_data

                    dpg.add_text("lr: ")
                    dpg.add_input_float(default_value=self.lr, width=150, format="%.6f", callback=callback_set_lr)

                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="bbox loss (modify in-bbox region)", default_value=self.enable_bbox_loss,
                                     callback=callback_toggle_mask_loss)

                    def callback_reset_bbox(sender, app_data):
                        if self.mesh is None:
                            return
                        vmin, vmax = self.mesh.aabb()
                        self.bbox_points = [vmin.detach().cpu().numpy().tolist(), vmax.detach().cpu().numpy().tolist()]
                        for i in range(2):
                            for j in range(3):
                                dpg.configure_item(f"_bbox_point_{i}_{j}", default_value=self.bbox_points[i][j])
                        self.need_update_overlay = True

                    dpg.add_button(label="reset bbox", tag="_button_reset_bbox", callback=callback_reset_bbox)
                    dpg.bind_item_theme("_button_reset_bbox", theme_button)

                    def callback_set_bbox_loss_weight(sender, app_data):
                        self.bbox_loss_weight = app_data

                    dpg.add_text("weight: ")
                    dpg.add_input_float(default_value=self.bbox_loss_weight, width=200, format="%.3f",
                                        callback=callback_set_bbox_loss_weight)

                with dpg.group(horizontal=True):
                    dpg.add_text("left-bottom: ")
                    dpg.add_slider_float(tag='_bbox_point_0_0', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[0][0], callback=callback_update_bbox_point,
                                         user_data=[0, 0])
                    dpg.add_slider_float(tag='_bbox_point_0_1', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[0][1], callback=callback_update_bbox_point,
                                         user_data=[0, 1])
                    dpg.add_slider_float(tag='_bbox_point_0_2', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[0][2], callback=callback_update_bbox_point,
                                         user_data=[0, 2])
                with dpg.group(horizontal=True):
                    dpg.add_text("right-top:   ")
                    dpg.add_slider_float(tag='_bbox_point_1_0', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[1][0], callback=callback_update_bbox_point,
                                         user_data=[1, 0])
                    dpg.add_slider_float(tag='_bbox_point_1_1', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[1][1], callback=callback_update_bbox_point,
                                         user_data=[1, 1])
                    dpg.add_slider_float(tag='_bbox_point_1_2', width=120, min_value=-1, max_value=1, format="%.3f",
                                         default_value=self.bbox_points[1][2], callback=callback_update_bbox_point,
                                         user_data=[1, 2])

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                    self.need_update_overlay = True

                dpg.add_combo(('albedo', 'depth', 'normal', 'lambertian'), label='mode', default_value=self.mode,
                              callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor",
                                   no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                    self.need_update_overlay = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)

                # light dir
                def callback_set_light_dir(sender, app_data, user_data):
                    self.light_dir[user_data] = app_data
                    self.need_update = True

                dpg.add_text("Plane Light Direction:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="theta", min_value=0, max_value=180, format="%.2f",
                                         default_value=self.light_dir[0], callback=callback_set_light_dir, user_data=0)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="phi", min_value=0, max_value=360, format="%.2f",
                                         default_value=self.light_dir[1], callback=callback_set_light_dir, user_data=1)

                # ambient ratio
                def callback_set_abm_ratio(sender, app_data):
                    self.ambient_ratio = app_data
                    self.need_update = True

                dpg.add_slider_float(label="ambient", min_value=0, max_value=1.0, format="%.5f",
                                     default_value=self.ambient_ratio, callback=callback_set_abm_ratio)

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True
            self.need_update_overlay = True

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True
            self.need_update_overlay = True

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
            self.need_update_overlay = True

        def callback_set_mouse_loc(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_keypoint_add(sender, app_data):

            if not dpg.is_item_focused("_primary_window") or self.buffer_rast is None:
                return

            # project mouse_loc to points_3d, if near to current, select it, else create new.
            rast = self.buffer_rast[0, int(self.mouse_loc[1]), int(self.mouse_loc[0])]

            # not hitting the mesh
            if rast[3] <= 0:
                return

            # use triangle-id and uv to interpolate the actual 3d point
            trig = self.mesh.f[rast[3].long() - 1]  # [3,]
            vert = self.mesh.v[trig.long()]  # [3, 3]
            uv = rast[:2]
            point_3d = (1 - uv[0] - uv[1]) * vert[0] + uv[0] * vert[1] + uv[1] * vert[2]
            point_3d = point_3d.detach().cpu().numpy()

            # decide if it's close to a current point, if so, just select it
            flag_mark_close = False
            if len(self.points_3d) > 0:
                cur_points = np.array(self.points_3d)
                dist = np.linalg.norm(cur_points - point_3d, axis=1)
                dist[~np.array(self.points_mask).astype(bool)] = 1e8  # ignore deleted points
                if np.min(dist) < 0.1:
                    # select the closest one
                    self.point_idx = np.argmin(dist)
                    flag_mark_close = True

            # else add a new point
            if not flag_mark_close:
                self.points_3d.append(point_3d.tolist())
                self.points_mask.append(True)
                self.points_3d_delta.append([0, 0, 0])
                # update UI
                _id = len(self.points_3d) - 1
                dpg.add_group(parent="_group_keypoints", tag=f"_group_keypoint_{_id}", horizontal=True)
                dpg.add_text(parent=f"_group_keypoint_{_id}",
                             default_value=f"{', '.join([f'{x:.3f}' for x in self.points_3d[_id]])} +")
                dpg.add_input_floatx(parent=f"_group_keypoint_{_id}", tag=f"_point_delta_{_id}", size=3, width=200,
                                     format="%.3f", on_enter=True, default_value=self.points_3d_delta[_id],
                                     callback=callback_update_keypoint_delta, user_data=_id)
                dpg.add_button(parent=f"_group_keypoint_{_id}", label="Del", callback=callback_delete_keypoint,
                               user_data=_id)
                self.point_idx = _id

            self.need_update_overlay = True

        def callback_keypoint_drag(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            if len(self.points_3d) == 0 or not self.points_mask[self.point_idx]:
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]

            delta = 0.00004 * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])

            self.points_3d_delta[self.point_idx][0] += delta[0]
            self.points_3d_delta[self.point_idx][1] += delta[1]
            self.points_3d_delta[self.point_idx][2] += delta[2]

            # update UI values
            dpg.configure_item(f"_point_delta_{self.point_idx}", default_value=self.points_3d_delta[self.point_idx])

            self.need_update_overlay = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_add)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_keypoint_drag)

        dpg.create_viewport(title='Drag Your 3DModel', width=self.W, height=self.H + (45 if os.name == 'nt' else 0),
                            resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists('LXGWWenKai-Regular.ttf'):
            with dpg.font_registry():
                with dpg.font('LXGWWenKai-Regular.ttf', 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()


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
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.

    c.outdir = opts.outdir
    c.fp32 = opts.fp32
    c.H = opts.height
    c.W = opts.width
    c.radius = opts.radius
    c.fovy = opts.fovy

    c.G_kwargs = dnnlib.EasyDict(
        class_name=None, z_dim=opts.latent_dim, w_dim=opts.latent_dim, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(
        class_name='training.networks_get3d.Discriminator', block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    c.inference_vis = opts.inference_vis
    # Training set.
    if opts.inference_vis:
        c.inference_to_generate_textured_mesh = opts.inference_to_generate_textured_mesh
        c.inference_save_interpolation = opts.inference_save_interpolation
        c.inference_compute_fid = opts.inference_compute_fid
        c.inference_generate_geo = opts.inference_generate_geo

    # c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, opt=opts)
    # if opts.cond and not c.training_set_kwargs.use_labels:
    #     raise click.ClickException('--cond=True requires labels specified in dataset.json')
    # c.training_set_kwargs.split = 'train' if opts.use_shapenet_split else 'all'
    # if opts.use_shapenet_split and opts.inference_vis:
    #     c.training_set_kwargs.split = 'test'
    # c.training_set_kwargs.use_labels = opts.cond
    # c.training_set_kwargs.xflip = False
    # Hyperparameters & settings.p
    c.G_kwargs.one_3d_generator = opts.one_3d_generator
    c.G_kwargs.n_implicit_layer = opts.n_implicit_layer
    c.G_kwargs.deformation_multiplier = opts.deformation_multiplier
    c.resume_pretrain = opts.resume_pretrain
    c.D_reg_interval = opts.d_reg_interval
    c.G_kwargs.use_style_mixing = opts.use_style_mixing
    c.G_kwargs.dmtet_scale = opts.dmtet_scale
    c.G_kwargs.feat_channel = opts.feat_channel
    c.G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
    c.G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
    c.G_kwargs.n_views = opts.n_views

    c.G_kwargs.render_type = opts.render_type
    c.G_kwargs.use_tri_plane = opts.use_tri_plane
    c.D_kwargs.data_camera_mode = opts.data_camera_mode
    c.D_kwargs.add_camera_cond = opts.add_camera_cond

    c.G_kwargs.tet_res = opts.tet_res

    c.G_kwargs.geometry_type = opts.geometry_type
    # c.num_gpus = opts.gpus
    # c.batch_size = opts.batch
    # c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    # c.G_kwargs.geo_pos_enc = opts.geo_pos_enc
    c.G_kwargs.data_camera_mode = opts.data_camera_mode
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax

    c.G_kwargs.mapping_kwargs.num_layers = 8

    c.D_kwargs.architecture = opts.d_architecture
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    # c.loss_kwargs.gamma_mask = opts.gamma if opts.gamma_mask == 0.0 else opts.gamma_mask
    # c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    # c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    # c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.network_snapshot_ticks = 200
    # Sanity checks.
    # if c.batch_size % c.num_gpus != 0:
    #     raise click.ClickException('--batch must be a multiple of --gpus')
    # if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
    #     raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    # if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
    #     raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    # if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
    #     raise click.ClickException(
    #         '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    # c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.networks_get3d.GeneratorDMTETMesh'
    c.loss_kwargs.style_mixing_prob = 0.9  # Enable style mixing regularization.
    c.loss_kwargs.pl_weight = 0.0  # Enable path length regularization.
    c.G_reg_interval = 4  # Enable lazy regularization for G.
    c.G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False
    
    # launch gui
    gui = GUI(c)
    gui.render()
if __name__ == "__main__":
    main()
