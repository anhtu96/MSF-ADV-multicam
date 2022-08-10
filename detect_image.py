import os
import cv2
import argparse
import matplotlib.pyplot as plt
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
from PIL import Image, ImageOps


class Yolov3(Darknet):
    def __init__(self, cfg_file):
        super().__init__(cfg_file)
        self.load_model_()

    def load_model_(self):
        namesfile = './pytorch/yolo_models/data_yolo/coco.names'
        self.class_names = load_class_names(namesfile)
        self.load_weights('./models/camera/yolov3/yolov3.weights')
        self.cuda()
        self.eval()

def predict_convert(image_var, model, class_names, reverse=False):
    # pred = model.get_spec_layer( (image_var - mean_var ) / std_dv_var, 0).max(1)[1]
    pred, _ = model(image_var)
    tmp = pred[0][0] + pred[1][0] + pred[2][0]
    for p in tmp:
        if p[-1] == 10:
            print(p)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', dest='img_path', default="./final_img_cam_2.png")
    parser.add_argument('--imsize', dest='imsize', type=int, default=416)
    args = parser.parse_args()
    background = cv2.imread(args.img_path)
    background = cv2.resize(background, (args.imsize, args.imsize))
    background = background[:, :, ::-1] / 255.0
    background = background.astype(np.float32)
    background = torch.tensor(background).permute(2, 0, 1).cuda().float()
    background = background.unsqueeze(0)
    model = Yolov3(f'pytorch/yolo_models/cfg/yolov3_{args.imsize}.cfg')
    preds, outputs = model(background)
    vis, boxes = predict_convert(background, model, model.class_names)
    plt.imsave('detection_result.png', vis)