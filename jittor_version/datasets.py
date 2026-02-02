#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import jittor as jt
from jittor import transform
import numpy as np
from PIL import Image
from jittor.dataset import Dataset

# 尝试导入已经修改为 Jittor 版本的 reorder
try:
    from designSeq import reorder
except ImportError:
    print("Warning: 'designSeq' not found. Using dummy reorder function.")
    def reorder(cls, box, mode, max_elem):
        return list(range(len(cls)))

# Jittor 不需要 unbind，直接切片
def box_xyxy_to_cxcywh(x):
    # 假设输入是 [N, 4]
    x0, y0, x1, y1 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)

# 继承自 jt.dataset的Dataset
class canvasLayout(Dataset):
    def __init__(self, inp_dir, sal_dir, json_path, max_elem=8, batch_size=1, shuffle=False):
        #【关键修改】：Jittor 的 Dataset 基类初始化，必须调用以启用多线程预取机制
        super().__init__()
        self.inp_dir = inp_dir
        self.sal_dir = sal_dir
        self.max_elem = max_elem
        
        print(f"Loading jittor annotations from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
            
        self.cls_map = {
            "title": 0, "subtitle": 0, "text": 0,
            "logo": 1, "underlay": 2
        }

        # 使用 jittor.transform
        self.transform = transform.Compose([
            #【关键修改】：Jittor 的 transform.Resize 算子只接受元组，不接受列表
            transform.Resize((350, 240)),
            transform.ToTensor()
        ])
        
        #【关键修改】：Jittor 弃用 DataLoader，直接在 Dataset 内部指定 BatchSize 和 Shuffle
        self.set_attrs(batch_size=batch_size, shuffle=shuffle)
        
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
            #【关键修改】：将显著图先转为 RGB 确保它是 3 维的 (H, W, 3)
            img_sal = Image.open(sal_path).convert("RGB") 
        except:
            img_inp = Image.new('RGB', (350, 240))
            # 修改同上
            img_sal = Image.new('RGB', (350, 240))

        # ---【关键修改】: 动态对齐显著图维度 ---
        # 1. 处理背景图
        img_inp_tensor = self.transform(img_inp) # 得到 [3, H, W]
        
        # 2. 获取背景图实际的 H 和 W (Jittor 张量维度是 [C, H, W])
        target_h, target_w = img_inp_tensor.shape[1], img_inp_tensor.shape[2]
        
        # 3. 手动处理显著图 (不使用 transform，避免 2D 转置报错)
        # PIL resize 使用 (Width, Height)
        img_sal_resized = img_sal.resize((target_w, target_h), Image.BILINEAR)
        img_sal_np = np.array(img_sal_resized)
        
        # 4. 转为 Jittor 张量并强制增加通道维，确保形状为 [1, H, W]
        img_sal_tensor = jt.array(img_sal_np).float() / 255.0
        if len(img_sal_tensor.shape) == 2:
            img_sal_tensor = img_sal_tensor.unsqueeze(0)
        
        # 5. 拼接成 4 通道 [4, H, W]
        cc = jt.concat([img_inp_tensor, img_sal_tensor], dim=0)
        
        label = np.zeros((self.max_elem, 2, 4), dtype=np.float32)
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
        
        if len(cls_list) > 0:
            # 转换成 numpy 供 reorder 逻辑使用
            box_np = np.array(box_list, dtype=np.float32)
            # 注意：reorder 函数内部现在应该处理 numpy 或 jt.Var
            order = reorder(cls_list, box_np, "xyxy", self.max_elem)
            
            for i in range(len(order)):
                idx_in_list = order[i]
                class_id = int(cls_list[idx_in_list])
                if class_id < 4:
                    label[i][0][class_id] = 1
                
                current_box = box_np[idx_in_list]
                # 坐标保护
                x0, y0, x1, y1 = current_box
                if x0 > x1: x0, x1 = x1, x0
                if y0 > y1: y0, y1 = y1, y0
                
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                w  = x1 - x0
                h  = y1 - y0
                
                label[i][1][0] = cx / orig_W
                label[i][1][1] = cy / orig_H
                label[i][1][2] = w  / orig_W
                label[i][1][3] = h  / orig_H

            for i in range(len(order), self.max_elem):
                label[i][0][3] = 1 # Padding 类
        
        #【关键修改】：返回 Numpy 格式，Jittor 会在组装 Batch 时自动将其转换为 jt.Var 避免手动转换
        return cc, label

class canvas(jt.dataset.Dataset):
    def __init__(self, bg_dir, sal_dir, json_path=None, batch_size=1):
        #【关键修改】：Jittor 的 Dataset 基类初始化，必须调用以启用多线程预取机制
        super().__init__()
        self.bg_dir = bg_dir
        self.sal_dir = sal_dir
        
        # 调试：打印路径
        if not os.path.exists(bg_dir):
            raise FileNotFoundError(f"Background directory not found: {bg_dir}")

        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.img_list = [os.path.basename(item["image"]) for item in data]
        else:
            self.img_list = [x for x in os.listdir(bg_dir) if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.img_list.sort()
            
        self.bg = [os.path.join(bg_dir, x) for x in self.img_list]
        self.transform = transform.Compose([
            # Jittor 的 transform.Resize 算子只接受元组，不接受列表
            transform.Resize((350, 240)),
            transform.ToTensor()
        ])
        #【关键修改】：Jittor 弃用 DataLoader，直接在 Dataset 内部指定 BatchSize 和 Shuffle
        self.set_attrs(batch_size=batch_size, shuffle=False)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        file_name = self.img_list[idx]
        bg_path = self.bg[idx]
        sal_path = os.path.join(self.sal_dir, file_name)
        
        try:
            img_bg = Image.open(bg_path).convert("RGB")
            # 显著图读入后保持为 "L" (灰度)
            img_sal = Image.open(sal_path).convert("L")
        except:
            # 这里的尺寸要和 transform 里的对应，但我们后面会动态调
            img_bg = Image.new('RGB', (240, 350))
            img_sal = Image.new('L', (240, 350))
        
        # ---【关键迁移逻辑】：推理阶段的背景图与显著图尺寸强制对齐及 4 通道手动拼接 ---
        
        # 1. 先处理背景图，获取其转换后的张量
        img_bg_tensor = self.transform(img_bg) # [3, H, W]
        
        # 2. 获取背景图张量的实际长宽
        # 这样可以 100% 保证显著图和背景图维度一致
        target_h = img_bg_tensor.shape[1]
        target_w = img_bg_tensor.shape[2]
        
        # 3. 手动处理显著图
        # 注意：PIL 的 resize 接收 (Width, Height)
        img_sal_resized = img_sal.resize((target_w, target_h), Image.BILINEAR)
        img_sal_np = np.array(img_sal_resized)
        
        # 4. 转换为张量并增加通道维度
        img_sal_tensor = jt.array(img_sal_np).float() / 255.0
        
        # 如果是 2 维 (H, W)，增加到 3 维 (1, H, W)
        if len(img_sal_tensor.shape) == 2:
            img_sal_tensor = img_sal_tensor.unsqueeze(0)
        
        # 5. 最终拼接 [3, H, W] + [1, H, W] = [4, H, W]
        return jt.concat([img_bg_tensor, img_sal_tensor], dim=0)