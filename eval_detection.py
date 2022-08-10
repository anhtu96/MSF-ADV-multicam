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
from PIL import Image
from dimension_reduction import reduce_dim, compress_image
import matplotlib.pyplot as plt

def eval_detection(args=None, save_gif=False):
    if args:
        attack_model = attack.attack_msf(args)
        attack_model.load_model_()
        attack_model.read_cali()
        attack_model.init_render()
        detected_center = 0
        detected_right = 0
        detected_both = 0
        undetected_both = 0
        undetected_right, undetected_center = 0, 0
        det_right_scenes, det_both_scenes, det_center_scenes = [], [], []
        x_range = np.arange(3, 7)
        y_range = [0, -1]
        # x_range = np.linspace()
        total_scenes = args.num_scenes * len(x_range) * len(y_range)
        for scene in tqdm.tqdm(range(args.num_scenes)):
            has_right, has_center = False, False
            # attack_model.args.rotations = [0, 0, 0]
            attack_model.args.frame_id = scene
            attack_model.get_rotation()
            attack_model.load_bg()
            if attack_model.dataset_name == 'argoverse':
                for xr in x_range:
                    for yr in y_range:
                        attack_model.args.position = [xr, yr]
                        attack_model.load_mesh()
                        has_right, has_center = False, False
                        for cam_num, cam_name in enumerate(attack_model.sensor_names):
                            if (args.multicam and (
                                    cam_name == 'ring_front_center' or cam_name == 'ring_front_right')) or \
                                    (not args.multicam and (cam_name == 'ring_front_center')):
                                im = attack_model.inject_obj_to_cam(cam_num, save=False)
                                im = cv2.resize(im, (attack_model.image_size, attack_model.image_size))
                                if args.reduce_dim:
                                    im = reduce_dim(im, args.svd_comps)
                                elif args.compress:
                                    im = compress_image(im, args.compress_colors)
                                    im = np.array(im) / 255.0
                                im = im.astype(np.float32)
                                im = torch.tensor(im).permute(2, 0, 1).cuda().float()
                                im = im.unsqueeze(0)
                                preds, outputs = attack_model.model(im)
                                boxes = utils_yolo.nms(preds[0][0] + preds[1][0] + preds[2][0], 0.4)
                                img_vis = (im[0].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                                pred_vis = plot_boxes(Image.fromarray(img_vis), boxes, class_names=attack_model.class_names,
                                                      print_result=False)
                                vis = np.array(pred_vis[0])
                                for i in range(len(boxes)):
                                    box = boxes[i]
                                    if attack_model.class_names[box[6]] in attack_model.args.obj_cls:
                                        if cam_name == 'ring_front_center':
                                            # print(attack_model.args.position)
                                            detected_center += 1
                                            has_center = True
                                            det_center_scenes.append((scene, xr, yr))
                                            # plt.imsave(f"center_{scene}_{xr}_{yr}.png", vis)
                                        elif cam_name == 'ring_front_right':
                                            # print(attack_model.args.position)
                                            detected_right += 1
                                            has_right = True
                                            det_right_scenes.append((scene, xr, yr))
                                            # plt.imsave(f"right_{scene}_{xr}_{yr}.png", vis)
                        if has_center and has_right:
                            detected_both += 1
                            det_both_scenes.append((scene, xr, yr))
                        elif not has_center and not has_right:
                            undetected_both += 1
                        if not has_center:
                            undetected_center += 1
                        if not has_right:
                            undetected_right += 1
                else:
                    im = attack_model.inject_obj_to_cam(2, save=False)
        asr_both = undetected_both / total_scenes
        asr_right = undetected_right / total_scenes
        asr_center = undetected_center / total_scenes
        print(f"Number of scenarios: {total_scenes}\n")
        print(f"Undetected center: {undetected_center}")
        print(f"Detected center: {detected_center}\n")
        if args.multicam:
            print(f"Undetected right: {undetected_right}")
            print(f"Detected right: {detected_right}\n")
        print(f"Undetected both: {undetected_both}\n")
        print(f"Attack success rate: {asr_both*100:.2f}%")
        print(f"Attack success rate center: {asr_center*100:.2f}%")
        if args.multicam:
            print(f"Attack success rate right: {asr_right*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--obj', dest='object', default="./object/object.ply")
    parser.add_argument('-obj_save', '--obj_save', dest='object_save', default="./object/object_adv")
    parser.add_argument('-c2r', '--cam_to_render', dest='cam_to_render', type=int, default=2)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, default='kitti')
    parser.add_argument('-data_path', '--data_path', dest='data_path', default='2011_09_26_drive_0002_sync')
    parser.add_argument('-num_scenes', '--num_scenes', dest='num_scenes', type=int, default=100)
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
    parser.add_argument('--num_degs', dest='num_degs', type=int, default=10)
    parser.add_argument('--init_degs', nargs='+', dest='init_degs', type=int, default=[0,0,0])
    parser.add_argument('--axis', dest='axis', type=str, default='xz')
    parser.add_argument('--obj_cls', nargs='+', dest='obj_cls', default=[])
    parser.add_argument('--reduce_dim', dest='reduce_dim', action='store_true')
    parser.add_argument('--svd_comps', dest='svd_comps', type=int, default=100)
    parser.add_argument('--compress', dest='compress', action='store_true')
    parser.add_argument('--compress_colors', dest='compress_colors', type=int, default=256)
    parser.add_argument('--multicam', dest='multicam', action='store_true')
    args = parser.parse_args()
    eval_detection(args)