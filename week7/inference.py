'''
/home/haeun/21winter/week7/s_models/0215_22:00/10_0_191118_234105_10_1_0_S_6_13_101101100011.png
: seg to class 완료!
지금 과제는 한글로 바꾸기
'''

# import argparse
# from doctest import OutputChecker
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import datasets
import torch
import torchvision.transforms as T

import warnings
warnings.filterwarnings(action='ignore')

from params import param_segmentation as s_params
from params import param_classification as c_params
from params import food_id
from transforms import classification_transforms as data_transform
import c_forInf as CI
from MaskRCNN import get_instance_segmentation_model


# 명령어: (haeun21w_1) haeun@gpu-server:~/21winter/week6/seg_from$ CUDA_VISIBLE_DEVICES=5 python tmp_inf.py
# device setting
device = torch.device("cuda")
print("... finish cuda setting")
# device = torch.device('cpu') 
# print("... finish cpu setting")

# load model (MASKRCNN)
# model_s_path = s_params['0215_13:30_best_model' # 이걸로 하면 뭔가 잘못됐어
model_s_path = s_params['model_1,2_pre_path']
print("... loading segmetation model")
model_s = get_instance_segmentation_model(num_classes=2)
model_s.to(device)
# model_s = model_s.cpu()
# model_s.load_state_dict(torch.load(model_s_path), map_location='cpu')
model_s.load_state_dict(torch.load(model_s_path))
model_s.eval()

# load model (ResNet)
data_dir = c_params['food_root']
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform['train'])
print("... loading classification model")
model_c = CI.load_checkpoint(device, dataset, c_params)
model_c.eval()

# load images and transform
# input image = real dataset_cycle1(easy)
inf_path = s_params['inf_path1']
print("... loading", end=' ')
img_list = sorted(os.listdir(inf_path)) 
if 'Thumbs.db' in img_list: img_list.remove("Thumbs.db")
transform = T.Compose([T.ToTensor(),
                        T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    ])
print("{} images".format(len(img_list)))

# visualization setting
out_path = s_params['output_dir']
out_path = os.path.join(out_path, '0218_04:39')
if not os.path.isdir(out_path):
    os.makedirs(out_path, exist_ok=True)
thres = float(0.5)
print("="*20)

print("+++ Start inference !")
for i, img_name in enumerate(img_list):
    if i<=10:
        print("... inference ({}/{}) _ {}".format(i+1, len(img_list), img_name))
        # [segmentation]
        # load and transform image
        img_file = os.path.join(inf_path, img_name) 
        img_data = Image.open(img_file).convert("RGB")

        # for segmentation
        img_tensor = transform(img_data) 
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_arr = np.array(img_data).astype(np.uint8)
        img_arr_c = np.array(img_data).astype(np.uint8)
        print("... finish image setting!")
        # save_name = os.path.join(out_path, '0.png')
        # cv2.imwrite(save_name, img_arr)
        # forward and post-process results
        pred_result = model_s(img_tensor, None)[0]
        pred_mask = pred_result['masks'].cpu().detach().numpy().transpose(0, 2, 3, 1)
        pred_mask[pred_mask >= 0.5] = 1
        pred_mask[pred_mask < 0.5] = 0
        pred_mask = np.repeat(pred_mask, 3, 3)
        pred_scores = pred_result['scores'].cpu().detach().numpy()
        pred_boxes = pred_result['boxes'].cpu().detach().numpy()
        # pred_labels = pred_result['labels']
        print("... finish box setting!")
        
        # draw predictions
        # print("[{} Scores]:".format(pred_scores.shape[0]), list(pred_scores))
        ids = np.where(pred_scores > thres)[0]
        colors = np.random.randint(0, 255, (len(ids), 3))
        for color_i, pred_i in enumerate(ids):
            # save_name = os.path.join(out_path, '1.png')
            # cv2.imwrite(save_name, img_arr)
            color = tuple(map(int, colors[color_i]))
            # draw segmentation
            mask = pred_mask[pred_i] 
            mask = mask * color
            img_arr = cv2.addWeighted(img_arr, 1, mask.astype(np.uint8), 0.5, 0)
            # save_name = os.path.join(out_path, '2.png')
            # cv2.imwrite(save_name, img_arr)
            # draw bbox and text
            x1, y1, x2, y2 = map(int, pred_boxes[pred_i])
            cv2.rectangle(img_arr, (x1, y1), (x2, y2), color, 2)
            # save_name = os.path.join(out_path, '3.png') # mask
            # cv2.imwrite(save_name, img_arr)
            # img_arr: numpy
            # 시작점 xy
            # 끝점 xy
            print("... finish drawing segmentation!")
            # add classification
            top = y2
            left = x1
            height = y2-y1
            width = x2-x1
            # c_image_crop = T.functional.crop(c_image, top, left, height, width)
            # for classification
            crop_img = img_arr_c[y1:y2, x1:x2]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            save_name = os.path.join(out_path, '_crop'+img_name)
            cv2.imwrite(save_name, crop_img)
            c_np_image = np.array(crop_img)/255
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            c_np_image = (c_np_image - mean) / std
            c_np_image = c_np_image.transpose((2, 0, 1))
            c_image = torch.Tensor(c_np_image).type(torch.cuda.FloatTensor)
            c_image = c_image.unsqueeze(0)
            output = model_c.forward(c_image)
            probabilities = torch.exp(output)
            top_probabilities, top_indices = probabilities.topk(3)

            # Convert to lists
            top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
            idx_to_class = {value: key for key, value in model_c.class_to_idx.items()}
            top = [idx_to_class[index] for index in top_indices]
            top = ' '.join(top)
            # top = food_id[top]
            print("... finish finding class!")
            
            # edit view
            vis_text = "Class:{}({:.2f})".format(top, pred_scores[pred_i])
            '''
            # font 여기부터 안되면 삭제
            font = ImageFont.truetype('/home/haeun/21winter/NanumFont/NanumGothic.ttf', 2)
            img_arr = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img_arr)
            draw.text((x1+5, y1+15), vis_text, font=font, fill=color)
            img_arr = np.array(img_arr)
            
            1. 탑3(o) -> 한줄에 하나씩 출력하는방법 ...
            2. 색칠한상태로 됐을수도 있으니까 크롭한거 저장해보기 ***
            3. cv2랑 numpy랑 인자가 다르니까 xy틀린건 없는지 확인해보기
            4. 크롭 안하고 numpy로 바로 슬라이싱해도 된다
            
            '''
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
            cv2.putText(img_arr, vis_text, (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # # save for debugging
            # cv2.imwrite("tmp_{}.png".format(color_i), img_arr)
            print("... finish editing 1 box!")
        # save visualized image
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        save_name = os.path.join(out_path, img_name)
        cv2.imwrite(save_name, img_arr)
        print("... finish editing 1 image!")