#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Inference Script for PosterLLaVa Dataset
Compatible with user's dataset.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 确保 dataset.py 在同一目录
from dataset import canvas 
from model import generator
import os
import numpy as np
from collections import OrderedDict

# --- 配置区域 ---
TEST_BG_PATH = "data/inpainted_1x"   
TEST_SAL_DIR = "data/saliency_map"
TEST_JSON_PATH = "data/test.json"     # 确保这个文件存在
CKPT_PATH = "output/DS-GAN-Epoch50.pth"   
OUTPUT_DIR = "output"
# ----------------

torch.manual_seed(0)
np.random.seed(0)
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
    
def random_init(batch, max_elem):
    # 根据你的 dataset.py，Padding 是第3类 (Index 3, 对应 cls_map 之外)
    # 这里的 coef 需要根据训练时的设定来，如果不确定，保持原样通常也可以跑
    coef = [0.1, 0.8, 1, 1] 
    cls_1 = torch.tensor(np.random.choice(4, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef)))
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)
    return init_layout

def test(G, testing_dl, fix_noise):
    G.eval()
    clses = []
    boxes = []
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Starting inference...")
    with torch.no_grad():
        for i, data in enumerate(testing_dl):
            # dataset.py 的 __getitem__ 只返回 cc (concat tensor)
            imgs = data.to(device)
            
            # 动态调整 noise
            current_batch_size = imgs.size(0)
            if current_batch_size != fix_noise.size(0):
                curr_noise = fix_noise[:current_batch_size].to(device)
            else:
                curr_noise = fix_noise.to(device)
            
            cls, box = G(imgs, curr_noise)
            
            clses.append(torch.argmax(cls.detach().cpu(), dim=-1, keepdim=True))
            boxes.append(box_cxcywh_to_xyxy(box.detach().cpu()))
            
            if i % 10 == 0:
                print(f"Processed batch {i}")

    if len(clses) > 0:
        clses = torch.concat(clses, dim=0).numpy()
        boxes = torch.concat(boxes, dim=0).numpy()
        
        torch.save(clses, os.path.join(OUTPUT_DIR, "clses-Epoch50.pt"))
        torch.save(boxes, os.path.join(OUTPUT_DIR, "boxes-Epoch50.pt"))
        print(f"Results saved to {OUTPUT_DIR}")
    else:
        print("No data processed.")
    
def main():
    test_batch_size = 4
    # 【重要】：你的 dataset.py 中 max_elem 默认为 8。
    # 如果你训练时没有修改这个参数，这里也应该是 8。
    max_elem = 10
    
    # 1. 加载数据集
    try:
        # 传入 json_path，canvas 类会自动解析
        testing_set = canvas(TEST_BG_PATH, TEST_SAL_DIR, TEST_JSON_PATH)
        testing_dl = DataLoader(testing_set, num_workers=4, batch_size=test_batch_size, shuffle=False)
        print(f"Testing set size: {len(testing_set)}")
        
        # 【关键修改】：直接利用 dataset 解析好的文件名列表
        # 这保证了 test_order.pt 和 DataLoader 的顺序绝对一致
        if hasattr(testing_set, 'img_list'):
            torch.save(testing_set.img_list, "test_order.pt")
            print(f"Saved test_order.pt with {len(testing_set.img_list)} names.")
        else:
            print("Error: testing_set has no attribute 'img_list'. Check dataset.py")
            return

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. 模型参数 (请确认 hidden_size = max_elem * 8 是否匹配你的训练配置)
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8, 
        "out_channels": 32, 
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    
    fix_noise = random_init(test_batch_size, max_elem)
    
    # 3. 加载模型
    try:
        G = generator(args_g)
        if not os.path.exists(CKPT_PATH):
            print(f"Checkpoint not found: {CKPT_PATH}")
            return
            
        print(f"Loading checkpoint from {CKPT_PATH}...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        G.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        # 如果报错 size mismatch，很可能是 max_elem 不对 (比如训练是10，这里是8，或者反过来)
        return
    
    if gpu:
        G = G.to(device)
        
    test(G, testing_dl, fix_noise)

if __name__ == "__main__":
    main()