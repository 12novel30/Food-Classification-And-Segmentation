import os
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from params import param_segmentation as s_params
from MaskRCNN import get_instance_segmentation_model

import warnings
warnings.filterwarnings(action='ignore')


# 명령어: (haeun21w_1) haeun@gpu-server:~/21winter/week6/seg_from$ CUDA_VISIBLE_DEVICES=5 python tmp_inf.py
device = torch.device("cuda")

# load model (MASKRCNN)
model_path = s_params['0215_13:30_best_model']
print("... loading model")
model = get_instance_segmentation_model(num_classes=2)
model.to(device)
# model = model.cpu()
# model.load_state_dict(torch.load(model_path), map_location='cpu')
model.load_state_dict(torch.load(model_path))
model.eval()

# load images and transform
# image = real dataset_cycle1(easy)
inf_path = s_params['inf_path']
print("... loading", end=' ')
img_list = sorted(os.listdir(inf_path)) 
if 'Thumbs.db' in img_list: img_list.remove("Thumbs.db")
transform = T.Compose([T.ToTensor(), #3.1
                        T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), #3.2
                    ])
print("{} images".format(len(img_list)))

# visualization setting
out_path= s_params['output_dir']
out_path = os.path.join(out_path, '0216_16:00')
if not os.path.isdir(out_path):
    os.makedirs(out_path, exist_ok=True)
thres = float(0.5)

print("+++ Start inference !")
for i, img_name in enumerate(img_list):
    print("... inference ({}/{}) _ {}".format(i+1, len(img_list), img_name))
    # load and transform image
    img_file = os.path.join(inf_path, img_name) #1
    img_data = Image.open(img_file).convert("RGB") #2
    # img_tensor = img_data.resize(img_resize, Img.BICUBIC)
    img_tensor = transform(img_data) #3
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_arr = np.array(img_data).astype(np.uint8)

    # forward and post-process results
    pred_result = model(img_tensor, None)[0]
    pred_mask = pred_result['masks'].cpu().detach().numpy().transpose(0, 2, 3, 1)
    pred_mask[pred_mask >= 0.5] = 1
    pred_mask[pred_mask < 0.5] = 0
    pred_mask = np.repeat(pred_mask, 3, 3)
    pred_scores = pred_result['scores'].cpu().detach().numpy()
    pred_boxes = pred_result['boxes'].cpu().detach().numpy()
    # pred_labels = pred_result['labels']

    # draw predictions
    # print("[{} Scores]:".format(pred_scores.shape[0]), list(pred_scores))
    ids = np.where(pred_scores > thres)[0]
    colors = np.random.randint(0, 255, (len(ids), 3))
    for color_i, pred_i in enumerate(ids):
        color = tuple(map(int, colors[color_i]))
        # draw segmentation
        mask = pred_mask[pred_i] 
        mask = mask * color
        img_arr = cv2.addWeighted(img_arr, 1, mask.astype(np.uint8), 0.5, 0)
        # draw bbox and text
        x1, y1, x2, y2 = map(int, pred_boxes[pred_i])
        cv2.rectangle(img_arr, (x1, y1), (x2, y2), color, 2)
        vis_text = "FOOD({:.2f})".format(pred_scores[pred_i])
        cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
        # cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # # save for debugging
        # cv2.imwrite("tmp_{}.png".format(color_i), img_arr)
    # save visualized image
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    save_name = os.path.join(out_path, img_name)
    cv2.imwrite(save_name, img_arr) 