import os
import numpy as np
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

from transforms import get_transform

class RealTrayDataset(Dataset):
    def __init__(self, mode, params):
        self.mode = mode # train mode or validation mode
        self.label_type = "unseen_food"
        # list setting
        data_roots = []
        self.rgb_list = []
        self.seg_list = []
        self.slot_label_list = []
        
        root_ = params['real_food_root']
        train1 = params['cycle2']
        val1 = params['cycle1']
        val2 = params['addition']
        last_image_ = params['image_']
        last_mask_ = params['mask_']
        if mode == "train":
            tmp_path = os.path.join(root_, train1)
            data_roots.append(tmp_path)
        if mode == "val":
            tmp_path = os.path.join(root_, val1)
            data_roots.append(tmp_path)
            tmp_path = os.path.join(root_, val2)
            data_roots.append(tmp_path)

        for data_root in data_roots:
            self.rgb_path = os.path.join(data_root, last_image_)
            # print(self.rgb_path)
            self.seg_path = os.path.join(data_root, last_mask_)
            rgb_list = list(sorted(glob.glob(os.path.join(self.rgb_path, '*.png'))))
            # print(len(rgb_list))
            if 'Thumbs.db' in rgb_list: rgb_list.remove("Thumbs.db")
            seg_list = list(sorted(glob.glob(os.path.join(self.seg_path, '*.mask'))))
            slot_label_list = list(sorted(glob.glob(os.path.join(self.seg_path, '*.txt'))))
            self.rgb_list += rgb_list
            self.seg_list += seg_list
            self.slot_label_list += slot_label_list
        assert len(self.rgb_list) > 0
        assert len(self.rgb_list) == len(self.seg_list)
        assert len(self.rgb_list) == len(self.slot_label_list)

        # load rgb transform
        self.transform_ver = "albumentation"
        self.rgb_transform, self.transform_ver = get_transform(self.transform_ver, self.mode)
        self.width = params['width']
        self.height = params['height']
        # labels in mask images
        self.slot_cat2label = {
                            '1_slot': 1,
                            '2_slot': 2,
                            '3_slot': 3,
                            '4_slot': 4,
                            '5_slot': 5
                        }
    
    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        # load rgb image
        rgb = Image.open(self.rgb_list[idx]).convert("RGB")
        rgb = rgb.resize((self.width, self.height), Image.BICUBIC)
        # load mask image
        make_arr = Image.open(self.seg_list[idx]).convert("L") #black
        make_arr = make_arr.resize((self.width, self.height), Image.NEAREST)
        make_arr = np.array(make_arr)
                
        # extrack masks
        slot_labels = open(self.slot_label_list[idx]).readlines()
        slot_id2category, slot_valid_ids = {}, []
        for slot_label in slot_labels:
            label, category = slot_label.split(' ') 
            category = category[:-1] if '\n' in category else category
            slot_id2category[int(label)] = category
            if category in self.slot_cat2label: slot_valid_ids.append(int(label))
        
        obj_ids = np.array([mask_id for mask_id in np.unique(make_arr) if mask_id in slot_valid_ids])
        masks = make_arr == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        temp_masks = []
        boxes = []
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1:
                continue
            temp_masks.append(masks[i])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  
        masks = np.array(temp_masks)
        labels = np.array(labels)
        boxes = np.array(boxes)

        # tensor format data
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # target
        target = {
            "image_id":image_id,
            "boxes":boxes,
            "area":area,
            "iscrowd":iscrowd,
            "masks":masks,
            "labels":labels
        }

        # RGB transform
        if self.transform_ver == "torchvision":
            rgb = self.rgb_transform(rgb)
        elif self.transform_ver == "albumentation":
            rgb = self.rgb_transform(image=np.array(rgb))['image']
        else:
            raise ValueError("Wrong transform version {}".format(self.transform_ver))

        return rgb, target
