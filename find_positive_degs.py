import numpy as np
import os
import argparse
import attack
import tqdm
import torch
import itertools
from pytorch.yolo_models import utils_yolo
import cv2
import imageio
from pytorch.yolo_models.utils_yolo import *

def find_positive_angles(num_degs, save_path, args=None, save_gif=False):
    x_degs = np.linspace(0, 360, num_degs)
    y_degs = np.linspace(0, 360, num_degs)
    z_degs = np.linspace(0, 360, num_degs)
    if not args:
        deg_list = [[0], [0], [0]]
    else:
        deg_list = [[args.init_degs[0]], [args.init_degs[1]], [args.init_degs[2]]]
    if 'x' in args.axis:
        deg_list[0] = x_degs
    if 'y' in args.axis:
        deg_list[1] = y_degs
    if 'z' in args.axis:
        deg_list[2] = z_degs
    deg_combinations = list(itertools.product(*deg_list))
    pos_degs = []
    if args:
        attack_model = attack.attack_msf(args)
        attack_model.load_model_()
        attack_model.read_cali()
        attack_model.init_render()
        attack_model.load_bg()
        for rot in tqdm.tqdm(deg_combinations):
            attack_model.args.rotations = rot
            attack_model.get_rotation()
            attack_model.load_mesh()
            if attack_model.dataset_name == 'argoverse':
                for cam_num, cam_name in enumerate(attack_model.sensor_names):
                    if cam_name == 'ring_front_center':
                        im = attack_model.inject_obj_to_cam(cam_num, save=False)
                        break
            else:
                im = attack_model.inject_obj_to_cam(2, save=False)
            im = cv2.resize(im, (attack_model.image_size, attack_model.image_size))
            im = im.astype(np.float32)
            im = torch.tensor(im).permute(2, 0, 1).cuda().float()
            im = im.unsqueeze(0)
            preds, outputs = attack_model.model(im)
            boxes = utils_yolo.nms(preds[0][0] + preds[1][0] + preds[2][0], 0.4)
            img_vis = (im[0].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            pred_vis = plot_boxes(Image.fromarray(img_vis), boxes, class_names=attack_model.class_names, print_result=False)
            vis = np.array(pred_vis[0])
            for i in range(len(boxes)):
                box = boxes[i]
                if attack_model.class_names[box[6]] in attack_model.args.obj_cls:
                    pos_degs.append(rot)
                    break
    np.savetxt(save_path, pos_degs, fmt="%.2f")
    return np.array(pos_degs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj', dest='object', default="./object/object.ply")
    parser.add_argument('-obj_save', '--obj_save', dest='object_save', default="./object/object_adv")
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
    parser.add_argument('--inject_only', dest='inject_only', action='store_true')
    parser.add_argument('--num_degs', dest='num_degs', type=int, default=10)
    parser.add_argument('--init_degs', nargs='+', dest='init_degs', type=int, default=[0,0,0])
    parser.add_argument('--axis', dest='axis', type=str, default='xz')
    parser.add_argument('--obj_cls', nargs='+', dest='obj_cls', default=[])
    args = parser.parse_args()
    deg_combinations = find_positive_angles(args.num_degs, 'pos_degrees.txt', args)
