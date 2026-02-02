#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 尝试导入 reorder
try:
    from designSeq import reorder
except ImportError:
    print("Warning: 'designSeq' not found. Using dummy reorder function.")
    def reorder(cls, box, mode, max_elem):
        return list(range(len(cls)))

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

class canvasLayout(Dataset):
    def __init__(self, inp_dir, sal_dir, json_path, max_elem=8):
        self.inp_dir = inp_dir
        self.sal_dir = sal_dir
        self.max_elem = max_elem
        
        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
            
        self.cls_map = {
            "title": 0,
            "subtitle": 0,
            "text": 0,
            "logo": 1,
            "underlay": 2
        }

        self.transform = transforms.Compose([
            transforms.Resize([350, 240]),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.data_list)
     
    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_path = item["image"]
        file_name = os.path.basename(full_path) 
        
        inp_path = os.path.join(self.inp_dir, file_name)
        sal_path = os.path.join(self.sal_dir, file_name)
        
        try:
            img_inp = Image.open(inp_path).convert("RGB")
            img_sal = Image.open(sal_path).convert("L") 
        except FileNotFoundError:
            # 简单的异常处理，防止单张图片缺失导致训练中断
            # print(f"Warning: Missing {file_name}, using blank.")
            img_inp = Image.new('RGB', (350, 240))
            img_sal = Image.new('L', (350, 240))

        img_inp_tensor = self.transform(img_inp)
        img_sal_tensor = self.transform(img_sal)
        
        cc = torch.concat([img_inp_tensor, img_sal_tensor])
         
        label = np.zeros((self.max_elem, 2, 4))
        orig_W = item.get("width", 513)
        orig_H = item.get("height", 750)
        
        elements = item.get("elements", [])
        cls_list = []
        box_list = [] 
        
        for elem in elements:
            e_type = elem["type"]
            e_bbox = elem["bbox"]
            cat_id = self.cls_map.get(e_type, 0)
            cls_list.append(cat_id)
            box_list.append(e_bbox)
            
        cls_list = cls_list[:self.max_elem]
        box_list = box_list[:self.max_elem]
        
        if len(cls_list) == 0:
             return cc, torch.tensor(label).float()

        box_tensor = torch.tensor(box_list, dtype=torch.float32)
        order = reorder(cls_list, box_tensor, "xyxy", self.max_elem)
        
        for i in range(len(order)):
            idx_in_list = order[i]
            class_id = int(cls_list[idx_in_list])
            if class_id < 4:
                label[i][0][class_id] = 1
            
            current_box = box_tensor[idx_in_list]
            if current_box[0] > current_box[2]: current_box[0], current_box[2] = current_box[2], current_box[0]
            if current_box[1] > current_box[3]: current_box[1], current_box[3] = current_box[3], current_box[1]
            
            cx = (current_box[0] + current_box[2]) / 2
            cy = (current_box[1] + current_box[3]) / 2
            w  = current_box[2] - current_box[0]
            h  = current_box[3] - current_box[1]
            
            label[i][1][0] = cx / orig_W
            label[i][1][1] = cy / orig_H
            label[i][1][2] = w  / orig_W
            label[i][1][3] = h  / orig_H

        for i in range(len(order), self.max_elem):
            # 这里的 3 对应 Padding 类别，需与 model.py 和 main.py 保持一致
            label[i][0][3] = 1
        
        return cc, torch.tensor(label).float()

class canvas(Dataset):
    def __init__(self, bg_dir, sal_dir, json_path=None):
        self.bg_dir = bg_dir
        self.sal_dir = sal_dir
        
        if json_path and os.path.exists(json_path):
            print(f"Loading test file list from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.img_list = [os.path.basename(item["image"]) for item in data]
        else:
            print(f"No JSON provided, loading all images from {bg_dir}...")
            self.img_list = [x for x in os.listdir(bg_dir) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.img_list.sort()
            
        # 【关键修复】：添加 self.bg 属性，存储完整路径，供 main.py 可视化使用
        self.bg = [os.path.join(bg_dir, x) for x in self.img_list]

        self.transform = transforms.Compose([
            transforms.Resize([350, 240]),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        file_name = self.img_list[idx]
        bg_path = self.bg[idx] # 直接使用 self.bg 中的路径
        sal_path = os.path.join(self.sal_dir, file_name)
        
        try:
            img_bg = Image.open(bg_path).convert("RGB")
            img_sal = Image.open(sal_path).convert("L")
        except FileNotFoundError:
            img_bg = Image.new('RGB', (350, 240))
            img_sal = Image.new('L', (350, 240))
        
        img_bg = self.transform(img_bg)
        img_sal = self.transform(img_sal)
        
        cc = torch.concat([img_bg, img_sal])
        return cc