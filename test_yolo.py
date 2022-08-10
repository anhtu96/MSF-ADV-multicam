import cv2
import torch
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet



if __name__ == "__main__":
    model = Darknet(f'./pytorch/yolo_models/cfg/yolov3_416.cfg')
    model.load_weights('./models/camera/yolov3/yolov3.weights')
    model.cuda().eval()
    img = cv2.imread('./injected_images/argoverse/cam_ring_front_center.png')
    img = cv2.resize(img, (416, 416))
    img = img.transpose(2, 0, 1) / 255.0
    img = torch.from_numpy(img).cuda().float().unsqueeze(0)
    final, outputs = model(img)
    num_pred = 0.0
    removed = 0.0
    threshold = 0.5
    for index, out in enumerate(outputs):
        num_anchor = out.shape[1] // 85
        print(num_anchor, out.shape)
        out = out.view(num_anchor, 80 + 5, out.shape[2], out.shape[3])
        cfs = torch.sigmoid(out[:, 4]).cuda()
        mask = (cfs >= threshold).type(torch.FloatTensor).cuda()
        num_pred += torch.numel(cfs)
        removed += torch.sum((cfs < threshold).type(torch.FloatTensor)).data.cpu().numpy()
        loss = torch.sum(mask * ((cfs - 0) ** 2 - (cfs-1) ** 2))
        print(loss)