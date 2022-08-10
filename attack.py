import sys

sys.path.append('/content/libs/caffe/python')

import neural_renderer as nr
from pytorch.renderer import nmr
import torch
import torch.autograd as autograd
import argparse
import cv2
from c2p_segmentation import *
import loss_LiDAR
import numpy as np
import cluster
import os
from xyz2grid import *
import render
from plyfile import *
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
import matplotlib.pyplot as plt
import utils.calibration
import pandas as pd


def read_cali(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        if 'R:' in line:
            rotation = line.split('R:')[-1]
        if 'T:' in line:
            translation = line.split('T:')[-1]
    tmp_r = rotation.split(' ')
    tmp_r.pop(0)
    tmp_r[-1] = tmp_r[-1].split('\n')[0]
    rota_matrix = []

    for i in range(3):
        tt = []
        for j in range(3):
            tt.append(float(tmp_r[i * 3 + j]))
        rota_matrix.append(tt)
    rota_matrix = np.array(rota_matrix)
    tmp_t = translation.split(' ')
    tmp_t.pop(0)
    tmp_t[-1] = tmp_t[-1].split('\n')[0]
    trans_matrix = [float(tmp_t[i]) for i in range(3)]
    trans_matrix = np.array(trans_matrix)
    return rota_matrix, trans_matrix


def get_rotation_from_list(rots):
    r = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    for rot in rots:
        r = np.dot(R.from_euler('xyz', rot, degrees=True).as_matrix(), r)
    return r


def predict_convert(image_var, model, class_names, reverse=False):
    # pred = model.get_spec_layer( (image_var - mean_var ) / std_dv_var, 0).max(1)[1]
    pred, _ = model(image_var)
    # print(np.array(pred).shape)
    boxes = []
    img_vis = []
    pred_vis = []
    vis = []
    i = 0
    boxes.append(nms(pred[0][i] + pred[1][i] + pred[2][i], 0.4))
    img_vis.append((image_var[i].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    pred_vis.append(plot_boxes(Image.fromarray(img_vis[i]), boxes[i], class_names=class_names))
    vis = np.array(pred_vis[i][0])
    return np.array(vis), np.array(boxes)


class attack_msf():
    def __init__(self, args):
        self.args = args
        self.num_pos = 1
        self.threshold = 0.4
        self.root_path = './data/'
        self.pclpath = 'pcd/'
        self.rotation = torch.tensor(np.array([[1., 0., 0.],
                                               [0., 0., -1.],
                                               [0., 1., 0.]]), dtype=torch.float)
        self.protofile = os.path.join('models/lidar', 'v' + self.args.lidar_model, 'deploy.prototxt')
        self.weightfile = os.path.join('models/lidar', 'v' + self.args.lidar_model, 'deploy.caffemodel')
        self.outputs = ['instance_pt', 'category_score', 'confidence_score',
                        'height_pt', 'heading_pt', 'class_score']
        self.esp = args.epsilon
        self.direction_val, self.dist_val = self.load_const_features('./data/features_1.out')
        self.dataset_name = self.args.dataset_name
        self.calib = {}
        self.position = {}
        self.get_rotation()
        self.orig_imsize = None
        self.image_size = self.args.imsize
        if self.dataset_name == 'argoverse':
            filenames = {}
            cam_path = os.path.join('./data/argoverse', self.args.data_path, 'sensors/cameras')
            lidar_path = os.path.join('./data/argoverse', self.args.data_path, 'sensors/lidar')
            lidar_bin_path = os.path.join('./data/argoverse', self.args.data_path, 'sensors/lidar_bin')
            if not os.path.exists(lidar_bin_path):
                os.makedirs(lidar_bin_path, exist_ok=True)
                for f in os.listdir(lidar_path):
                    ft = pd.read_feather(os.path.join(lidar_path, f))
                    ft = ft.to_numpy()[:, :4].astype(np.float32)
                    ft.tofile(os.path.join(lidar_bin_path, f.replace('.feather', '.bin')))
            for cam_name in os.listdir(cam_path):
                filenames[cam_name] = sorted(os.listdir(os.path.join(cam_path, cam_name)))
            filenames['lidar'] = sorted(os.listdir(lidar_bin_path))
            self.lidar_filename = filenames['lidar'][self.args.frame_id]
            self.cam_filename = {cam_name: filenames[cam_name][self.args.frame_id*2] for cam_name in os.listdir(cam_path)}
            self.timestamp_ms = round(int(self.lidar_filename.split('.bin')[0]) / 10**6)

    def init_render(self, image_size=608):
        self.renderer = nr.Renderer(image_size=self.image_size, camera_mode='look_at',
                                    anti_aliasing=False, light_direction=(0, 0, 0))
        exr = cv2.imread('./data/dog.exr', cv2.IMREAD_UNCHANGED)
        self.renderer.light_direction = [1, 3, 1]

        ld, lc, ac = nmr.lighting_from_envmap(exr)
        self.renderer.light_direction = ld
        self.renderer.light_color = lc
        self.renderer.ambient_color = ac
        self.renderer.camera_direction = [0, 0, 1]

    def get_rotation(self):
        self.rotation_list = [[self.args.rotations[0], 0, 0], [0, self.args.rotations[1], 0],
                              [0, 0, self.args.rotations[2]]]
        self.rotation_list_inv = [[0, 0, -self.args.rotations[2]], [0, -self.args.rotations[1], 0],
                                  [-self.args.rotations[0], 0, 0]]
        rotation_matrix = get_rotation_from_list(self.rotation_list)
        self.lidar_rotation = torch.tensor(rotation_matrix, dtype=torch.float).cuda()

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def load_const_features(self, fname):

        print("Loading direction, dist")
        features_filename = fname

        features = np.loadtxt(features_filename)
        features = np.swapaxes(features, 0, 1)
        features = np.reshape(features, (1, 512, 512, 8))

        direction = np.reshape(features[:, :, :, 3], (1, 512, 512, 1))
        dist = np.reshape(features[:, :, :, 6], (1, 512, 512, 1))
        return torch.tensor(direction).cuda().float(), torch.tensor(dist).cuda().float()

    def model_val_lidar(self, protofile, weightfile):
        net = CaffeNet(protofile, phase='TEST')
        # torch.cuda.set_device(0)
        net.cuda()
        net.load_weights(weightfile)
        net.set_train_outputs(outputs)
        net.set_eval_outputs(outputs)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def load_LiDAR_model(self, ):
        self.LiDAR_model = generatePytorch(self.protofile, self.weightfile).cuda()
        self.LiDAR_model_val = self.model_val_lidar(self.protofile, self.weightfile)

    def load_model_(self):

        namesfile = './pytorch/yolo_models/data_yolo/coco.names'
        self.class_names = load_class_names(namesfile)
        single_model = Darknet(f'./pytorch/yolo_models/cfg/yolov3_{self.image_size}.cfg')
        single_model.load_weights('./models/camera/yolov3/yolov3.weights')
        model = single_model
        self.model = model.cuda()
        self.model.eval()

    def load_pc_mesh(self):
        if self.dataset_name == 'kitti':
            PCL_path = os.path.join('./data/kitti', self.args.data_path, f'velodyne_points/data/{self.args.frame_id:010}.bin')
        elif self.dataset_name == 'argoverse':
            PCL_path = os.path.join('./data/argoverse', self.args.data_path, f'sensors/lidar_bin/{self.lidar_filename}')
        # loading ray_direction and distance for the background pcd
        self.PCL = loadPCL(PCL_path, True)
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)

    def inject_obj_to_cam(self, cam=2, injected_path='injected_images/kitti', size=608, save=True):
        if self.dataset_name == 'argoverse':
            injected_path = 'injected_images/argoverse'
            self.T_velo_to_cam = utils.calibration.create_transform_matrix(
                self.rota_matrix[cam].T, self.rota_matrix[cam].T.dot(-self.trans_matrix[cam]))
        os.makedirs(injected_path, exist_ok=True)
        if self.dataset_name == 'argoverse':
            cam_name = self.sensor_names[cam]
            if 'lidar' in cam_name:
                return
            cam_files = os.listdir(os.path.join('./data/argoverse/', self.args.data_path, 'sensors/cameras', cam_name))
            path = os.path.join('./data/argoverse/', self.args.data_path, 'sensors/cameras', cam_name, self.cam_filename[cam_name])
        elif self.dataset_name == 'kitti':
            path = os.path.join('./data/kitti', f"{self.args.data_path}", f"image_{cam:02}/data", f"{self.args.frame_id:010}.png")
            cam_name = str(cam)
        background = cv2.imread(path)
        self.orig_imsize = background.shape
        background = cv2.resize(background, (self.image_size, self.image_size))
        background = background[:, :, ::-1] / 255.0
        background = background.astype(np.float32)

        pts_3d_velo = self.object_v.detach().cpu().numpy()
        pts_3d_ref = np.dot(self.cart2hom(pts_3d_velo), np.transpose(self.T_velo_to_cam))
        if self.dataset_name == 'kitti':

            pts_3d_rect = np.transpose(np.dot(self.calib[f'R_rect_{cam:02}'], np.transpose(pts_3d_ref)))
            pts_3d_rect += self.calib[f'T_{cam:02}']
            c_v_c = torch.Tensor(pts_3d_rect).cuda()
        elif self.dataset_name == 'argoverse':
            c_v_c = torch.Tensor(pts_3d_ref[:,:3]).cuda()
        # c_v_c[:,0] *= self.image_size / self.orig_imsize[1]
        # c_v_c[:,1] *= self.image_size / self.orig_imsize[0]
        u_offset = 0
        v_offset = 0
        image_tensor = self.renderer.render(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0), self.object_t.unsqueeze(0))[
            0].cuda()
        image_tensor = torch.flip(image_tensor, dims=[2])
        mask_tensor = self.renderer.render_silhouettes(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0)).cuda()
        mask_tensor = torch.flip(mask_tensor, dims=[1])
        background_tensor = torch.from_numpy(background.transpose(2, 0, 1)).cuda()
        fg_mask_tensor = torch.zeros(background_tensor.size())
        new_mask_tensor = mask_tensor.repeat(3, 1, 1)
        fg_mask_tensor[:, u_offset: u_offset + self.image_size, v_offset: v_offset + self.image_size] = new_mask_tensor
        fg_mask_tensor = fg_mask_tensor.bool().cuda()
        new_mask_tensor = new_mask_tensor.bool().cuda()
        background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))
        final_image = torch.clamp(background_tensor.float(), 0, 1)[None]
        im = final_image[0].detach().cpu().numpy().transpose(1, 2, 0)
        if save:
            plt.imsave(os.path.join(injected_path, f'cam_{cam_name}.png'), im)
        return im

    def load_mesh(self):
        # z_of = -1.73 + self.args.obj_scale / 2.
        z_of = 0
        self.position['x'] = self.args.position[0]
        self.position['y'] = self.args.position[1]
        self.position['z'] = z_of
        self.position['r'] = self.args.obj_scale
        plydata = PlyData.read(self.args.object)
        x = torch.FloatTensor(plydata['vertex']['x']) * self.args.obj_scale
        y = torch.FloatTensor(plydata['vertex']['y']) * self.args.obj_scale
        z = torch.FloatTensor(plydata['vertex']['z']) * self.args.obj_scale
        self.object_v = torch.stack([x, y, z], dim=1).cuda()

        self.object_f = plydata['face'].data['vertex_indices']
        self.object_f = torch.tensor(np.vstack(self.object_f).astype(np.float32)).cuda()

        rotation = self.lidar_rotation.cuda()
        self.object_v = self.object_v.cuda()
        self.object_v = self.object_v.permute(1, 0)
        self.object_v = torch.matmul(rotation, self.object_v)
        self.object_v = self.object_v.permute(1, 0)
        self.object_v[:, 0] += self.args.position[0]
        self.object_v[:, 1] += self.args.position[1]
        self.object_v[:, 2] += z_of

        self.object_ori = self.object_v.clone()

        # pts_3d_velo = self.object_v.detach().cpu().numpy()
        # pts_3d_ref = np.dot(self.cart2hom(pts_3d_velo), np.transpose(self.T_velo_to_cam))
        # pts_3d_rect = np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
        # camera_v = self.object_v.clone()
        # camera_v = self.object_v.clone().cpu().numpy()
        # camera_v = self.cart2hom(camera_v)
        # camera_v = camera_v.transpose(1, 0)

        # r, t = torch.tensor(self.rota_matrix).cuda().float(), torch.tensor(self.trans_matrix).cuda().float()
        # r_c = R.from_euler('zxy', [0, 180, 180], degrees=True)
        # camera_v = torch.matmul(r, camera_v)
        # camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
        # camera_v = torch.matmul(camera_rotation, camera_v)

        # camera_v = camera_v.permute(1, 0)
        # camera_v += t
        # c_v_c = camera_v.cuda()
        # c_v_c = torch.Tensor(pts_3d_rect).cuda()
        # print(c_v_c)
        # c_v_c = torch.Tensor(pts_3d_rect).cuda()
        # self.vn, idxs = self.set_neighbor_graph(self.object_f, c_v_c)
        # self.vn_tensor = torch.Tensor(self.vn).view(-1).cuda().long()
        # self.idxs_tensor = torch.Tensor(idxs.copy()).cuda().long()

        # self.object_t = torch.tensor(self.object_v.new_ones(self.object_f.shape[0], 1, 1, 1, 3)).cuda()
        self.object_t = self.object_v.new_ones(self.object_f.shape[0], 1, 1, 1, 3).clone().detach().cuda()
        # color red
        self.object_t[:, :, :, :, 0] = self.args.colors[0]
        self.object_t[:, :, :, :, 1] = self.args.colors[1]
        self.object_t[:, :, :, :, 2] = self.args.colors[2]
        self.mean_gt = self.object_ori.mean(0).data.cpu().clone().numpy()

        # u_offset = 0
        # v_offset = 0
        # image_tensor = self.renderer.render(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0), self.object_t.unsqueeze(0))[0].cuda()
        # image_tensor = torch.flip(image_tensor, dims=[2])
        # mask_tensor = self.renderer.render_silhouettes(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0)).cuda()
        # mask_tensor = torch.flip(mask_tensor, dims=[1])
        # background_tensor = torch.from_numpy(self.background.transpose(2, 0, 1)).cuda()
        # fg_mask_tensor = torch.zeros(background_tensor.size())
        # new_mask_tensor = mask_tensor.repeat(3, 1, 1)
        # fg_mask_tensor[:, u_offset: u_offset + self.image_size, v_offset: v_offset + self.image_size] = new_mask_tensor
        # fg_mask_tensor = fg_mask_tensor.bool().cuda()
        # new_mask_tensor = new_mask_tensor.bool().cuda()
        # background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))
        # final_image = torch.clamp(background_tensor.float(), 0, 1)[None]
        # im = final_image[0].detach().cpu().numpy().transpose(1, 2, 0)
        # im[im<0] = 0
        # im[im>1] = 1
        # plt.imsave('final_img.png', im)

    def set_learning_rate(self, optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def tv_loss_(self, image, ori_image):
        noise = image - ori_image
        loss = torch.mean(torch.abs(noise[:, :, :, :-1] - noise[:, :, :, 1:])) + torch.mean(
            torch.abs(noise[:, :, :-1, :] - noise[:, :, 1:, :]))
        return loss

    def load_bg(self, h=608, w=608):
        if self.dataset_name == 'argoverse':
            path = os.path.join('./data/argoverse/', self.args.data_path, 'sensors/cameras', 'ring_front_center', self.cam_filename['ring_front_center'])
        elif self.dataset_name == 'kitti':
            path = os.path.join('./data/kitti', f"{self.args.data_path}", f"image_02/data", f"{self.args.frame_id:010}.png")
            cam_name = str(2)
        # path = './data/argoverse/b87683ae-14c5-321f-8af3-623e7bafc3a7/sensors/cameras/ring_front_center/315972334349927215.jpg'
        # path = os.path.join(self.args.data_path,
        #                     f'image_{self.args.cam_to_render:02}/data/{self.args.frame_id:010}.png')
        background = cv2.imread(path)
        self.orig_imsize = background.shape
        background = cv2.resize(background, (self.image_size, self.image_size))
        background = background[:, :, ::-1] / 255.0
        self.background = background.astype(np.float32)

    def compute_total_variation_loss(self, img1, img2):
        diff = img1 - img2
        tv_h = ((diff[:, :, 1:, :] - diff[:, :, :-1, :]).pow(2)).sum()
        tv_w = ((diff[:, 1:, :, :] - diff[:, :-1, :, :]).pow(2)).sum()
        return tv_h + tv_w

    def l2_loss(self, desk_t, desk_v, ori_desk_t, ori_desk_v):
        t_loss = torch.nn.functional.mse_loss(desk_t, ori_desk_t)
        v_loss = torch.nn.functional.mse_loss(desk_v, ori_desk_v)
        return v_loss, t_loss

    def rendering_img(self):
        if self.args.inject_only:
            if self.dataset_name == 'argoverse':
                injected_path = f'injected_images/argoverse'
                for cam in range(len(self.sensor_names)):
                    self.T_velo_to_cam = utils.calibration.create_transform_matrix(
                        self.rota_matrix[cam].T, self.rota_matrix[cam].T.dot(-self.trans_matrix[cam]))
                    self.inject_obj_to_cam(cam, injected_path)
            elif self.dataset_name == 'kitti':
                injected_path = f'injected_images/kitti'
                for cam in range(4):
                    self.inject_obj_to_cam(cam, injected_path)
            return
        ppath = os.path.join(self.args.data_path, f'velodyne_points/data/{self.args.frame_id:10}.bin')
        u_offset = 0
        v_offset = 0

        lr = 0.005
        best_it = 1e10
        num_class = 80
        threshold = 0.25
        batch_size = 1

        self.object_v.requires_grad = True
        bx = self.object_v.clone().detach().requires_grad_()
        sample_diff = np.random.uniform(-0.001, 0.001, self.object_v.shape)
        sample_diff = torch.tensor(sample_diff).cuda().float()
        sample_diff.clamp_(-args.epsilon, args.epsilon)
        self.object_v.data = sample_diff + bx
        iteration = self.args.iteration
        if self.args.opt == 'Adam':
            from torch.optim import Adam
            opt = Adam([self.object_v], lr=lr, amsgrad=True)
        if self.args.random_shift:
            angle_shifts = [0, 15, -15]
        else:
            angle_shifts = [0]

        for a_s in angle_shifts:
            print(f'Object shifted to {a_s:.2f} degrees')
            self.rotation_list[2][2] += a_s
            self.rotation_list_inv[0][2] -= a_s
            rotation_matrix = R.from_euler('xyz', [0, 0, a_s], degrees=True).as_matrix()
            rotation = torch.tensor(rotation_matrix, dtype=torch.float).cuda()
            self.object_f = self.object_f.cuda()
            self.i_final = self.i_final.cuda()

            self.object_v = self.object_v.cuda()
            self.object_v = torch.matmul(rotation, self.object_v.T).T
            for it in range(iteration):
                if it % 200 == 0:
                    lr = lr / 10.0
                l_c_c_ori = self.object_ori

                # self.object_v = self.random_obj(self.object_v)
                adv_total_loss = None

                # shape (122999, 4)
                point_cloud = render.render(self.ray_direction, self.length, self.object_v, self.object_f, self.i_final)
                if self.args.lidar_model == '2.0':
                    grid = xyzi2grid_v2(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3],
                                        X_RES=512, Y_RES=512)
                    featureM = gridi2feature_v2(grid, self.direction_val, self.dist_val)
                elif self.args.lidar_model == '5.5':
                    grid = xyz2grid(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], X_RES=672, Y_RES=672)
                    featureM = grid2feature_v2(grid)
                outputPytorch = self.LiDAR_model(featureM)
                lossValue, loss_object, loss_distance, loss_center, loss_z = loss_LiDAR.lossRenderAttack(outputPytorch,
                                                                                                         self.object_v,
                                                                                                         self.object_ori,
                                                                                                         self.object_f,
                                                                                                         0.05)

                # camera_v = self.object_v.clone()
                # camera_v = camera_v.permute(1, 0)

                # r, t = torch.tensor(self.rota_matrix).cuda().float(), torch.tensor(self.trans_matrix).cuda().float()
                # r_c = R.from_euler('zxy', [0, 0, 0], degrees=True)
                # camera_v = torch.matmul(r, camera_v)
                # camera_v = camera_v.permute(1, 0)
                # camera_v = camera_v.permute(1, 0)
                # camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
                # camera_v = torch.matmul(camera_rotation, camera_v)
                # camera_v = camera_v.permute(1, 0)
                # camera_v += t
                # self.object_ori = self.object_v.clone()
                pts_3d_velo = self.object_v.clone()
                pts_3d_hom = torch.cat((pts_3d_velo, torch.ones((pts_3d_velo.shape[0], 1)).cuda().float()), dim=1)
                if self.dataset_name == 'kitti':
                    pts_3d_ref = torch.matmul(pts_3d_hom, torch.tensor(self.T_velo_to_cam).T.cuda().float())
                    pts_3d_rect = torch.matmul(torch.tensor(self.calib[f'R_rect_{self.args.cam_to_render:02}']).cuda().float(),
                                               pts_3d_ref.T).T
                    pts_3d_rect += torch.tensor(self.calib[f'T_{self.args.cam_to_render:02}']).cuda().float()
                    c_v_c = pts_3d_rect
                elif self.dataset_name == 'argoverse':
                    self.T_velo_to_cam = utils.calibration.create_transform_matrix(
                        self.rota_matrix[0].T, self.rota_matrix[0].T.dot(-self.trans_matrix[0]))
                    pts_3d_ref = torch.matmul(pts_3d_hom, torch.tensor(self.T_velo_to_cam).T.cuda().float())
                    c_v_c = pts_3d_ref[:,:3]
                image_tensor = \
                self.renderer.render(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0), self.object_t.unsqueeze(0))[0].cuda()
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = self.renderer.render_silhouettes(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0)).cuda()
                mask_tensor = torch.flip(mask_tensor, dims=[1])
                background_tensor = torch.from_numpy(self.background.transpose(2, 0, 1)).cuda()
                fg_mask_tensor = torch.zeros(background_tensor.size())
                new_mask_tensor = mask_tensor.repeat(3, 1, 1)
                fg_mask_tensor[:, u_offset: u_offset + self.image_size,
                v_offset: v_offset + self.image_size] = new_mask_tensor
                fg_mask_tensor = fg_mask_tensor.bool().cuda()
                new_mask_tensor = new_mask_tensor.bool().cuda()
                background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))

                final_image = torch.clamp(background_tensor.float(), 0, 1)[None]

                final, outputs = self.model(final_image)

                num_pred = 0.0
                removed = 0.0
                for index, out in enumerate(outputs):
                    num_anchor = out.shape[1] // (num_class + 5)
                    out = out.view(batch_size * num_anchor, num_class + 5, out.shape[2], out.shape[3])
                    cfs = torch.sigmoid(out[:, 4]).cuda()
                    mask = (cfs >= threshold).type(torch.FloatTensor).cuda()
                    num_pred += torch.numel(cfs)
                    removed += torch.sum((cfs < threshold).type(torch.FloatTensor)).data.cpu().numpy()

                    loss = torch.sum(mask * ((cfs - 0) ** 2 - (1 - cfs) ** 2))

                    if adv_total_loss is None:
                        adv_total_loss = loss
                    else:
                        adv_total_loss += loss
                total_loss = 12 * (F.relu(torch.clamp(adv_total_loss, min=0) - 0.01) / 5.0)
                total_loss += lossValue
                if best_it > total_loss.data.cpu() or it == 0:
                    best_it = total_loss.data.cpu().clone()
                    best_vertex = self.object_v.data.cpu().clone()
                    best_final_img = final_image.data.cpu().clone()
                    best_out = outputs.copy()
                    best_face = self.object_f.data.cpu().clone()
                    best_out_lidar = outputPytorch[:]
                    pc_ = point_cloud[:, :3].cpu().detach().numpy()

                if it % self.args.print_every == 0:
                    print('Iteration {} of {}: Loss={}'.format(it, iteration, total_loss.data.cpu().numpy()))
                self.object_v = self.object_v.cuda()

                if self.args.opt == "Adam":
                    opt.zero_grad()
                    total_loss.backward(retain_graph=True)
                    opt.step()
                else:
                    pgd_grad = autograd.grad([total_loss.sum()], [self.object_v])[0]
                    with torch.no_grad():
                        loss_grad_sign = pgd_grad.sign()
                        self.object_v.data.add_(alpha=-lr, other=loss_grad_sign)
                        diff = self.object_v - bx
                        diff.clamp_(-self.esp, self.esp)
                        self.object_v.data = diff + bx
                    del pgd_grad
                    del diff

                if it < iteration - 1:
                    del total_loss
                    del featureM
                    del grid
                    del point_cloud

        print('best iter: {}'.format(best_it))
        diff = self.object_v - bx
        vertice = best_vertex.numpy()
        face = best_face.numpy()
        pp = ppath.split('/')[-1].split('.bin')[0]
        self.object_v = torch.tensor(best_vertex).cuda()
        if self.args.inject_only:
            if self.dataset_name == 'argoverse':
                injected_path = f'injected_images/argoverse'
                for cam in range(len(self.sensor_names)):
                    self.T_velo_to_cam = utils.calibration.create_transform_matrix(
                        self.rota_matrix[cam].T, self.rota_matrix[cam].T.dot(-self.trans_matrix[cam]))
                    self.inject_obj_to_cam(cam, injected_path)
                return
            elif self.dataset_name == 'kitti':
                injected_path = f'injected_images/kitti'
                for cam in range(4):
                    self.inject_obj_to_cam(cam, injected_path)


        # Recover the original object's angle
        rotation_matrix_inv = get_rotation_from_list(self.rotation_list_inv)
        vertice_orig_angle = vertice.copy()
        vertice_orig_angle[:, 0] -= self.position['x']
        vertice_orig_angle[:, 1] -= self.position['y']
        vertice_orig_angle[:, 2] -= self.position['z']
        vertice_orig_angle = np.dot(rotation_matrix_inv, vertice_orig_angle.T).T
        # render.savemesh(self.args.object, self.args.object_save + 'lidar' + '_v2.ply', vertice, face, r=0.33)
        self.savemesh(self.args.object, self.args.object_save, vertice_orig_angle,
                      r=self.position['r'])
        # render.savemesh(self.args.object, self.args.object_save + pp + '_v2.ply', vertice, face, r=0.33)

        print('x range: ', vertice[:, 0].max() - vertice[:, 0].min())
        print('y range: ', vertice[:, 1].max() - vertice[:, 1].min())
        print('z range: ', vertice[:, 2].max() - vertice[:, 2].min())
        ######################
        PCLConverted = mapPointToGrid(pc_)

        print('------------  Pytorch Output ------------')
        obj, label_map = cluster.cluster(best_out_lidar[1].cpu().detach().numpy(),
                                         best_out_lidar[2].cpu().detach().numpy(),
                                         best_out_lidar[3].cpu().detach().numpy(),
                                         best_out_lidar[0].cpu().detach().numpy(),
                                         best_out_lidar[5].cpu().detach().numpy())

        # obstacle, cluster_id_list = twod2threed(obj, label_map, self.PCL, PCLConverted)
        self.pc_save = pc_
        self.best_final_img = best_final_img.numpy()
        self.best_vertex = best_vertex.numpy()
        self.benign = bx.clone().data.cpu().numpy()
        with open('final_pc.bin', 'wb') as f:
            f.write(point_cloud.cpu().detach().numpy())
        plt.imsave('final_image.png', self.best_final_img[0].transpose(1, 2, 0))

    def savemesh(self, path_r, path_w, vet, r):
        plydata = PlyData.read(path_r)
        plydata['vertex']['x'] = vet[:, 0] / r
        plydata['vertex']['y'] = vet[:, 1] / r
        plydata['vertex']['z'] = vet[:, 2] / r
        if os.path.exists(path_w):
            os.remove(path_w)
        plydata.write(path_w)
        return

    def set_neighbor_graph(self, f, vn, degree=1):
        max_len = 0
        face = f.cpu().data.numpy()
        vn = vn.data.cpu().tolist()
        for i in range(len(face)):
            v1, v2, v3 = face[i]
            for v in [v1, v2, v3]:
                vn[v].append(v2)
                vn[v].append(v3)
                vn[v].append(v1)

        # two degree
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
        for de in range(degree - 1):
            vn2 = [[] for _ in range(len(vn))]
            for i in range(len(vn)):
                for item in vn[i]:
                    vn2[i].extend(vn[item])

            for i in range(len(vn2)):
                vn2[i] = list(set(vn2[i]))
            vn = vn2
        max_len = 0
        len_matrix = []
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
            len_matrix.append(len(vn[i]))

        idxs = np.argsort(len_matrix)[::-1][:len(len_matrix) // 1]
        max_len = len_matrix[idxs[0]]
        print("max_len: ", max_len)

        vns = np.zeros((len(idxs), max_len))
        # for i in range( len(vn)):
        for i0, i in enumerate(idxs):
            for j in range(max_len):
                if j < len(vn[i]):
                    vns[i0, j] = vn[i][j]
                else:
                    vns[i0, j] = i
        return vns, idxs

    def read_cali(self, add_cam=2):
        if self.dataset_name == 'argoverse':
            path = os.path.join('./data/argoverse', self.args.data_path, 'calibration/egovehicle_SE3_sensor.feather')
            self.rota_matrix, self.trans_matrix, self.sensor_names = utils.calibration.read_from_feather(path)
            self.K = utils.calibration.get_camera_intrinsic_matrix(os.path.join('./data/argoverse', self.args.data_path, 'calibration/intrinsics.feather'))
        elif self.dataset_name == 'kitti':
            path = os.path.join('./data/kitti', 'cali.txt')
            file1 = open(path, 'r')
            Lines = file1.readlines()
            for line in Lines:
                if 'R:' in line:
                    rotation = line.split('R:')[-1].strip().split(' ')
                if 'T:' in line:
                    translation = line.split('T:')[-1].strip().split(' ')
            self.rota_matrix = np.array([float(i) for i in rotation]).reshape(3, 3)
            self.trans_matrix = np.array([float(i) for i in translation])
            self.T_velo_to_cam = np.hstack([self.rota_matrix, self.trans_matrix.reshape(-1, 1)])
            # read additional cam calibration
            file2 = open(os.path.join('./data/kitti', 'calib_cam_to_cam.txt'))
            Lines = file2.readlines()
            for line in Lines:
                for cam in range(4):
                    if f'R_rect_{cam:02}:' in line:
                        R_rect = line.split(f'R_rect_{cam:02}:')[-1].strip().split(' ')
                        self.calib[f'R_rect_{cam:02}'] = np.array([float(i) for i in R_rect]).reshape(3, 3)
                    elif f'P_rect_{cam:02}:' in line:
                        P_rect = line.split(f'P_rect_{cam:02}:')[-1].strip().split(' ')
                        self.calib[f'P_rect_{cam:02}'] = np.array([float(i) for i in P_rect]).reshape(3, 4)
                    elif f'T_{cam:02}:' in line:
                        T = line.split(f'T_{cam:02}:')[-1].strip().split(' ')
                        self.calib[f'T_{cam:02}'] = np.array([float(i) for i in T]).reshape(1, 3)
                    elif f'K_{cam:02}:' in line:
                        K = line.split(f'K_{cam:02}:')[-1].strip().split(' ')
                        self.calib[f'K_{cam:02}'] = np.array([float(i) for i in K]).reshape(3, 3)
                    elif f'D_{cam:02}:' in line:
                        D = line.split(f'D_{cam:02}:')[-1].strip().split(' ')
                        self.calib[f'D_{cam:02}'] = np.array([float(i) for i in D]).reshape(1, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj', dest='object', default="./object/object.ply")
    parser.add_argument('-obj_save', '--obj_save', dest='object_save', default="./object/object_adv.ply")
    parser.add_argument('-c2r', '--cam_to_render', dest='cam_to_render', type=int, default=2)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='kitti')
    parser.add_argument('-data_path', '--data_path', dest='data_path', default='2011_09_26_drive_0002_sync')
    parser.add_argument('-frame_id', '--frame_id', dest='frame_id', type=int, default=0)
    parser.add_argument('--rotations', nargs='+', dest='rotations', type=int, default=[0,0,0])
    parser.add_argument('--colors', nargs='+', dest='colors', type=float, default=[0.3, 0.3, 0.3])
    parser.add_argument('--obj_scale', dest='obj_scale', type=float, default=1)
    parser.add_argument('--position', nargs='+', dest='position', type=float, default=[10, 2])
    # parser.add_argument('-lidar', '--lidar', dest='lidar')
    # parser.add_argument('-cam', '--cam', dest='cam')
    parser.add_argument('-o', '--opt', dest='opt', default="pgd")
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)
    parser.add_argument('--lidar_model', dest='lidar_model', type=str, default='5.5')
    parser.add_argument('--imsize', dest='imsize', type=int, default=416)
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=1000)
    parser.add_argument('-print_every', '--print_every', dest='print_every', type=int, default=10)
    parser.add_argument('--random_shift', dest='random_shift', action='store_true')
    parser.add_argument('--inject_only', dest='inject_only', action='store_true')
    args = parser.parse_args()

    obj = attack_msf(args)
    if not args.inject_only:
        obj.load_model_()
        obj.load_LiDAR_model()
    obj.read_cali()
    obj.init_render()

    obj.load_bg()
    obj.load_mesh()
    obj.load_pc_mesh()
    obj.rendering_img()