#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified for PosterLLaVa Dataset integration
"""

import os
# 【新增】解决 OpenMP 冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import copy
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import log
from collections import OrderedDict
import matplotlib.pyplot as plt

# --- 配置区域 (请根据你的实际路径修改) ---
# 评估时使用的标准分辨率 (PosterLayout 原论文标准，建议保持不变以维持指标计算的阈值有效性)
EVAL_W = 513
EVAL_H = 750
# 配置要测试的 Epoch 列表
EPOCH_LIST = [50, 100, 150, 200, 250, 300] 
PLOT_SAVE_PATH = "output/metrics_plot.png" # 折线图保存名

# 数据集路径
IMG_DIR = "data/inpainted_1x"   # 背景图路径 (用于计算 Readability 和画图)
SAL_DIR = "data/saliency_map"   # 显著图路径
# -------------------------------------

gpu = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")

def draw_box(img, elems, elems2):
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    
    # 【修改点】: 更新颜色映射以匹配 dataset.py 的类别 (0:Text, 1:Logo, 2:Underlay)
    # 原始: {1: 'green', 2: 'red', 3: 'orange'}
    cls_color_dict = {0: 'green', 1: 'red', 2: 'orange'}
    
    # 获取图片宽高！！！
    W, H = img.size
    
    for cls, box in elems:
        # 确保 cls 是整数
        c_idx = int(cls[0]) if isinstance(cls, np.ndarray) or isinstance(cls, list) else int(cls)
        
        # 过滤过大框！！！
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        area_ratio = (box_w * box_h) / (W * H)
        # 如果面积超过 90%，则不绘制（认为它是背景噪音）！！！
        if area_ratio > 0.9:
            continue
        
        if c_idx in cls_color_dict:
            draw_ol.rectangle(tuple(box), fill=None, outline=cls_color_dict[c_idx], width=5)
    
    # 对填充层进行排序绘制
    s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        c_idx = int(cls[0]) if isinstance(cls, np.ndarray) or isinstance(cls, list) else int(cls)
        
        # 过滤过大框！！！
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        area_ratio = (box_w * box_h) / (W * H)
        # 如果面积超过 90%，则不绘制（认为它是背景噪音）！！！
        if area_ratio > 0.9:
            continue
        
        if c_idx in cls_color_dict:
            draw_f.rectangle(tuple(box), fill=cls_color_dict[c_idx])
            
    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.3))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    
    return drawn

def cvt_pilcv(img, req='pil2cv', color_code=None):
    if req == 'pil2cv':
        if color_code == None:
            color_code = cv2.COLOR_RGB2BGR
        dst = cv2.cvtColor(np.asarray(img), color_code)
    elif req == 'cv2pil':
        if color_code == None:
            color_code = cv2.COLOR_BGR2RGB
        dst = Image.fromarray(cv2.cvtColor(img, color_code))
    return dst

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    # 避免除零
    max_val = np.max(grad_xy)
    if max_val > 0:
        grad_xy = grad_xy / max_val * 255
    img_g_xy = Image.fromarray(grad_xy).convert('L')
    return img_g_xy

def metrics_iou(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2
    
    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter =  min(yr_1, yr_2) - max(yl_1, yl_2)
 
    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0
 
    if (a_1 + a_2 - a_inter) <= 0: return 0
    return a_inter / (a_1 + a_2 - a_inter)

def metrics_inter_oneside(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2
    
    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter =  min(yr_1, yr_2) - max(yl_1, yl_2)
 
    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0
 
    if a_2 <= 0: return 0
    return a_inter / a_2

def metrics_val(img_size, clses, boxes):
    """
    The ratio of non-empty layouts.
    Higher is better.
    """
    w, h = img_size
    
    total_elem = 0
    empty_elem = 0
    
    for cls, box in zip(clses, boxes):
        # 这里假设 mask 逻辑：只要 cls 有值就是有效
        # 注意：在 dataset.py 中我们用 label[i][0][cls_idx] = 1
        # 这里的 clses 可能是 argmax 后的结果，或者是 ground truth 的 class index
        # 如果 clses 是 one-hot, 需要先 argmax。这里假设输入已经是 index
        
        mask = (cls > -1).reshape(-1) # 假设 padding 是特定负值，或者自行调整
        # 如果你的 output/clses-Epoch300.pt 存储的是 padding=0? 
        # 原逻辑 cls > 0，意味着 0 是 invalid/padding。
        # 但现在 0 是 Text。
        # 通常模型输出会带有一个 Padding Class (比如 3 或 4)
        # 这里假设有效元素是 0, 1, 2。
        
        # 暂时保留原逻辑结构，但在调用前需确认 clses 的格式
        mask_box = box # 简化：假设 boxes 里的 0,0,0,0 是无效的
        
        # 简单判定：如果 box 面积 > 0 则有效
        valid_indices = []
        for k in range(len(box)):
             if (box[k][2] - box[k][0]) > 1 and (box[k][3] - box[k][1]) > 1:
                 valid_indices.append(k)
        
        total_elem += len(valid_indices)
        
        for k in valid_indices:
            mb = box[k]
            xl, yl, xr, yr = mb
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(EVAL_W, xr)
            yr = min(EVAL_H, yr)
            # 这个阈值是原论文设定的，基于 513x750 分辨率
            if abs((xr - xl) * (yr - yl)) < (EVAL_W/100.0) * (EVAL_H/100.0) * 10: 
                empty_elem += 1
                
    if total_elem == 0: return 0
    return 1 - empty_elem / total_elem

def getRidOfInvalid(img_size, clses, boxes):
    w, h = img_size
    # 复制一份以免修改原数据
    new_clses = copy.deepcopy(clses)
    
    for i, (cls, box) in enumerate(zip(new_clses, boxes)):
        for j, b in enumerate(box):
            xl, yl, xr, yr = b
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(EVAL_W, xr)
            yr = min(EVAL_H, yr)
            if abs((xr - xl) * (yr - yl)) < (EVAL_W/100.0) * (EVAL_H/100.0) * 10:
                # 将无效的类别标记为 -1 或其他 Padding ID
                # 原代码是置 0，但现在 0 是 Text。
                # 假设 padding ID 是 3 (或更高)
                pass 
                # 这里仅仅是原来的逻辑，暂时不动，避免破坏 tensor 结构
    return new_clses

def metrics_uti(img_names, clses, boxes):
    """
    Utility: Non-saliency / Content Occlusion
    """
    metrics = 0
    for idx, name in enumerate(img_names):
        # 【修改点】: 只读取单张显著图，路径改为 SAL_DIR
        sal_path = os.path.join(SAL_DIR, name) # 假设文件名一致
        try:
            pic = np.array(Image.open(sal_path).convert("L").resize((EVAL_W, EVAL_H))) / 255
        except:
            # 容错
            pic = np.zeros((EVAL_H, EVAL_W))
            
        c_pic = np.ones_like(pic) - pic
        
        cal_mask = np.zeros_like(pic)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        # 筛选有效 Box (长宽大于0)
        valid_idx = np.where((box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1]))[0]
        mask_box = box[valid_idx]
        
        for mb in mask_box:
            xl, yl, xr, yr = mb
            # 边界保护
            xl, yl = max(0, xl), max(0, yl)
            xr, yr = min(EVAL_W, xr), min(EVAL_H, yr)
            cal_mask[yl:yr, xl:xr] = 1
        
        total_not_sal = np.sum(c_pic)
        total_utils = np.sum(c_pic * cal_mask)
        
        if total_not_sal > 0:
            metrics += (total_utils / total_not_sal)
            
    return metrics / len(img_names)

def metrics_rea(img_names, clses, boxes):
    '''
    Readability: Text on flat regions (low gradient).
    Lower is better.
    '''
    metrics = 0
    for idx, name in enumerate(img_names):
        # 【修改点】: 读取背景图
        img_path = os.path.join(IMG_DIR, name)
        try:
            pic = Image.open(img_path).convert("RGB").resize((EVAL_W, EVAL_H))
        except:
            continue # 跳过找不到的图片

        img_g_xy = np.array(img_to_g_xy(pic)) / 255
        cal_mask = np.zeros_like(img_g_xy)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        # 【修改点】: 类别映射调整
        # Text = 0
        # Underlay = 2
        text_indices = np.where(cls == 0)[0]
        deco_indices = np.where(cls == 2)[0]
        
        text_box = box[text_indices]
        deco_box = box[deco_indices]
        
        # 1. 标记 Text 区域为 1
        for mb in text_box:
            xl, yl, xr, yr = mb
            xl, yl = max(0, xl), max(0, yl)
            xr, yr = min(EVAL_W, xr), min(EVAL_H, yr)
            cal_mask[yl:yr, xl:xr] = 1
            
        # 2. 标记 Underlay 区域为 0 (Underlay 改善了可读性，所以不计算梯度惩罚)
        for mb in deco_box:
            xl, yl, xr, yr = mb
            xl, yl = max(0, xl), max(0, yl)
            xr, yr = min(EVAL_W, xr), min(EVAL_H, yr)
            cal_mask[yl:yr, xl:xr] = 0
        
        total_area = np.sum(cal_mask)
        total_grad = np.sum(img_g_xy[cal_mask == 1])
        
        if total_area > 0:
            metrics += (total_grad / total_area)
            
    return metrics / len(img_names)

def metrics_ove(clses, boxes):
    """
    Ratio of overlapping area (excluding underlay).
    Lower is better.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ove = 0
        # 【修改点】: 排除 Underlay (2) 和 Padding/Invalid
        # 假设 valid class 是 0, 1, 2. 
        # 我们只计算 Text(0) 和 Logo(1) 之间的重叠
        mask = np.where((cls == 0) | (cls == 1))[0]
        
        mask_box = box[mask]
        n = len(mask_box)
        if n > 1:
            for i in range(n):
                bb1 = mask_box[i]
                for j in range(i + 1, n):
                    bb2 = mask_box[j]
                    ove += metrics_iou(bb1, bb2)
            metrics += ove / (n * (n-1) / 2) # Normalize by pairs? 原代码是 / n
        else:
            metrics += 0
            
    # 原代码 metrics += ove / n 逻辑有点奇怪，若是 N个物体互斥，应为 0。
    # 这里保持原代码风格，但在 mask 上做了适配。
    return metrics / len(clses)

def metrics_und_l(clses, boxes):
    """
    Underlay Effectiveness (Large overlap).
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        # 【修改点】: Underlay = 2, Other = 0 or 1
        mask_deco = np.where(cls == 2)[0]
        mask_other = np.where((cls == 0) | (cls == 1))[0]
        
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        
        n1 = len(box_deco)
        n2 = len(box_other)
        
        if n1 > 0 and n2 > 0:
            avali += 1
            for i in range(n1):
                max_ios = 0
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    ios = metrics_inter_oneside(bb1, bb2)
                    max_ios = max(max_ios, ios)
                und += max_ios
            metrics += und / n1
            
    if avali > 0:
        return metrics / avali
    return 0

def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    
    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    c3 = xr_1 >= xr_2 # 原代码可能有笔误 xr_2 >= xr_2? 修正为 xr_1 >= xr_2 (bb1 包含 bb2)
    c4 = yr_1 >= yr_2
 
    return c1 and c2 and c3 and c4

def metrics_und_s(clses, boxes):
    """
    Underlay Effectiveness (Strict containment).
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        # 【修改点】: Underlay = 2
        mask_deco = np.where(cls == 2)[0]
        mask_other = np.where((cls == 0) | (cls == 1))[0]
        
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        
        n1 = len(box_deco)
        n2 = len(box_other)
        
        if n1 > 0 and n2 > 0:
            avali += 1
            for i in range(n1):
                bb1 = box_deco[i] # Underlay
                for j in range(n2):
                    bb2 = box_other[j] # Text/Logo
                    # 检查 Underlay 是否包含 Text
                    # 原函数名 is_contain(bb1, bb2) -> bb1 contains bb2
                    # 注意原代码 is_contain 实现可能有 bug (c3 = xr_2 >= xr_2 是恒成立)
                    # 这里暂且沿用，但在上面修正了 is_contain 的逻辑
                    if is_contain(bb1, bb2):
                        und += 1
                        break
            metrics += und / n1
            
    if avali > 0:
        return metrics / avali
    return 0

def ali_g(x):
    if 1 - x <= 0: return 0 # 防止 math domain error
    return -log(1 - x, 10)

def ali_delta(xs):
    n = len(xs)
    min_delta = np.inf
    if n < 2: return 0
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(xs[i] - xs[j])
            min_delta = min(min_delta, delta)
    return min_delta

def metrics_ali(clses, boxes):
    """
    Alignment.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ali = 0
        # 【修改点】: 只计算 Text(0) 和 Logo(1) 的对齐
        mask = np.where((cls == 0) | (cls == 1))[0]
        mask_box = box[mask]
        
        theda = []
        for mb in mask_box:
            pos = copy.deepcopy(mb)
            # 归一化回 0-1 用于计算对齐，因为 ali_g 是基于 0-1 的
            pos[0] /= EVAL_W
            pos[2] /= EVAL_W
            pos[1] /= EVAL_H
            pos[3] /= EVAL_H
            # [xl, yl, xc, yc, xr, yr]
            theda.append([pos[0], pos[1], (pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2, pos[2], pos[3]])
        
        theda = np.array(theda)
        if theda.shape[0] <= 1:
            continue
        
        # 计算对齐
        g_vals = []
        for j in range(6): # 6个对齐轴
            xys = theda[:, j]
            delta = ali_delta(xys)
            g_vals.append(ali_g(delta))
        
        # 取最好的对齐方式
        metrics += min(g_vals)

    return metrics / len(clses)

def metrics_occ(img_names, clses, boxes):
    '''
    Occlusion (Saliency) - similar to Utilization but reversed context usually.
    PosterLayout usage: Average saliency of pixels covered by elements.
    Lower is better (don't cover salient parts).
    '''
    metrics = 0
    for idx, name in enumerate(img_names):
        # 【修改点】: 单张显著图
        sal_path = os.path.join(SAL_DIR, name)
        try:
            pic = np.array(Image.open(sal_path).convert("L").resize((EVAL_W, EVAL_H))) / 255
        except:
            pic = np.zeros((EVAL_H, EVAL_W))
            
        cal_mask = np.zeros_like(pic)
        
        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)
        
        # 同样排除无效box
        valid_idx = np.where((box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1]))[0]
        mask_box = box[valid_idx]
        
        for mb in mask_box:
            xl, yl, xr, yr = mb
            xl, yl = max(0, xl), max(0, yl)
            xr, yr = min(EVAL_W, xr), min(EVAL_H, yr)
            cal_mask[yl:yr, xl:xr] = 1
        
        total_area = np.sum(cal_mask)
        total_sal = np.sum(pic[cal_mask == 1])
        
        if total_area > 0:
            metrics += (total_sal / total_area)
            
    return metrics / len(img_names)
        
def save_figs(names, clses, boxes, save_dir):
    try:
        os.makedirs(save_dir)
    except:
        pass
    for idx, name in enumerate(names):
        img_path = os.path.join(IMG_DIR, name)
        try:
            pic = Image.open(img_path).convert("RGB").resize((EVAL_W, EVAL_H))
            cls = np.array(clses[idx], dtype=int)
            box = np.array(boxes[idx], dtype=int)
            drawn = draw_box(pic, zip(cls, box), zip(cls, box))
            # 保存文件名处理
            save_name = os.path.splitext(name)[0] + "_vis.png"
            drawn.save(os.path.join(save_dir, save_name))
        except Exception as e:
            print(f"Could not save fig for {name}: {e}")
            
# --- 新增函数：绘制折线图 ---
def plot_metrics(epochs, history, save_path):
    """
    绘制并保存指标随 Epoch 变化的折线图。
    history: dict, key是指标名, value是对应 epochs 的列表
    """
    metrics_names = list(history.keys())
    num_metrics = len(metrics_names)
    
    # 动态计算子图布局
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, metric in enumerate(metrics_names):
        plt.subplot(rows, cols, i + 1)
        values = history[metric]
        plt.plot(epochs, values, marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 标出最后一个点的值
        if len(values) > 0:
            plt.annotate(f"{values[-1]:.4f}", 
                         (epochs[-1], values[-1]), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Metrics plot saved to {save_path}")

def main():
    # 1. 加载文件名列表
    try:
        names = torch.load("test_order.pt", weights_only=False)
        print(f"Number of test samples: {len(names)}")
    except FileNotFoundError:
        print("Error: 'test_order.pt' not found. Run inference first.")
        return
    
    # 初始化历史记录字典
    history = {
        'Validity': [], 'Overlap': [], 'Alignment': [],
        'Underlay_L': [], 'Underlay_S': [],
        'Utility': [], 'Occlusion': [], 'Readability': []
    }
    
    valid_epochs = [] # 记录实际跑成功的 epoch

    # 2. 循环处理每个 Epoch
    for epoch in EPOCH_LIST:
        print(f"\n{'='*20} Processing Epoch {epoch} {'='*20}")
        
        # 构造文件名
        cls_path = f"output/clses-Epoch{epoch}.pt"
        box_path = f"output/boxes-Epoch{epoch}.pt"
        
        # 检查文件是否存在
        if not (os.path.exists(cls_path) and os.path.exists(box_path)):
            print(f"Skipping Epoch {epoch}: Files not found (clses/boxes).")
            continue
            
        try:
            clses = torch.load(cls_path, weights_only=False)
            boxes = torch.load(box_path, weights_only=False)
        except Exception as e:
            print(f"Error loading Epoch {epoch}: {e}")
            continue

        valid_epochs.append(epoch)

        '''# --- 强制修正类别逻辑 (Heuristic Fix) ---
        batch_size, num_elems, _ = boxes.shape
        for b in range(batch_size):
            for i in range(num_elems):
                w = boxes[b, i, 2]
                h = boxes[b, i, 3]
                area = w * h
                
                # 规则: 大面积 -> Underlay (2)
                #if area > 0.5: ！！！
                    #clses[b, i, 0] = 2 ！！！
                
                # 规则: 极小近似方形 -> Logo (1)
                if area < 0.02 and 0.8 < w/h < 1.2:
                    clses[b, i, 0] = 1
        # ----------------------------------------------------'''

        # --- 数据转换与缩放 ---
        # 拷贝数据防止修改原 Tensor
        if hasattr(boxes, 'clone'): boxes_scaled = boxes.clone()
        else: boxes_scaled = boxes.copy()
            
        boxes_scaled[:, :, ::2] *= EVAL_W
        boxes_scaled[:, :, 1::2] *= EVAL_H
        
        # 转 Numpy
        clses_np = clses.cpu().numpy() if torch.is_tensor(clses) else clses
        boxes_np = boxes_scaled.cpu().numpy() if torch.is_tensor(boxes_scaled) else boxes_scaled
        
        # --- 【关键功能保留】计算并打印指标 ---
        print("Calculating metrics...")
        
        val = metrics_val((EVAL_W, EVAL_H), clses_np, boxes_np)
        ove = metrics_ove(clses_np, boxes_np)
        ali = metrics_ali(clses_np, boxes_np)
        und_l = metrics_und_l(clses_np, boxes_np)
        und_s = metrics_und_s(clses_np, boxes_np)
        uti = metrics_uti(names, clses_np, boxes_np)
        occ = metrics_occ(names, clses_np, boxes_np)
        rea = metrics_rea(names, clses_np, boxes_np)

        # 存入历史
        history['Validity'].append(val)
        history['Overlap'].append(ove)
        history['Alignment'].append(ali)
        history['Underlay_L'].append(und_l)
        history['Underlay_S'].append(und_s)
        history['Utility'].append(uti)
        history['Occlusion'].append(occ)
        history['Readability'].append(rea)
        
        # 打印到控制台 (和你原来的输出一样)
        print(f"metrics_val (Validity): {val}")
        print(f"metrics_ove (Overlap): {ove}")
        print(f"metrics_ali (Alignment): {ali}")
        print(f"metrics_und_l (Underlay Large): {und_l}")
        print(f"metrics_und_s (Underlay Strict): {und_s}")
        print(f"metrics_uti (Utility): {uti}")
        print(f"metrics_occ (Occlusion): {occ}")
        print(f"metrics_rea (Readability): {rea}")

        # --- 【关键功能保留】保存可视化图片 ---
        # 为了不让程序跑太慢，建议只在最后一个 Epoch 保存图片
        # 如果你想所有Epoch都保存，去掉 if 判断即可
        if epoch == EPOCH_LIST[-1]: 
            print(f"Saving visualizations for Epoch {epoch}...")
            save_dir = f"output/result_plot_epoch{epoch}/"
            save_figs(names, clses_np, boxes_np, save_dir)
            print("Visualizations saved.")

    # 3. 绘制所有 Epoch 的趋势图
    if len(valid_epochs) > 0:
        plot_metrics(valid_epochs, history, PLOT_SAVE_PATH)
    else:
        print("No valid epochs found to plot.")

'''def main():
    no = 1
    save_dir = f"output/result_plot/"
    
    # 1. 加载测试文件列表
    # 确保你在 dataset.py 的测试模式下运行过，生成了 test_order.pt
    try:
        names = torch.load("test_order.pt", weights_only=False)
        print("Number of test samples:", len(names))
    except FileNotFoundError:
        print("Error: 'test_order.pt' not found. Run dataset.py or inference first.")
        return

    # 2. 加载模型预测结果
    # 请确保这些路径指向你实际生成的 .pt 文件
    try:
        # 【修改点2 & 3】添加 weights_only=False
        clses_path = "output/clses-Epoch300.pt"
        boxes_path = "output/boxes-Epoch300.pt"
        
        clses = torch.load(clses_path, weights_only=False) # 预测类别
        boxes = torch.load(boxes_path, weights_only=False) # 预测框 (0-1归一化)
    except FileNotFoundError:
        print("Error: Output pt files not found. Check 'output/' directory.")
        return
    
    # --- 强行修正类别逻辑 ---
    # 如果一个框的面积 > 画布面积的 15%，且它在最底层（列表后面），强制将其视为 Underlay (2)
    # 如果一个框特别小且接近方形，视为 Logo (1)
    # 这是一个简单的补救措施
    
    batch_size, num_elems, _ = boxes.shape
    for b in range(batch_size):
        for i in range(num_elems):
            # boxes 是 cx, cy, w, h (0-1归一化)
            w = boxes[b, i, 2]
            h = boxes[b, i, 3]
            area = w * h
        
            # 规则1: 大面积 -> Underlay
            if area > 0.5: 
                clses[b, i, 0] = 2 # 强制改为 Underlay
            
            # 规则2: 极小面积 -> Logo (可选)
            if area < 0.02 and 0.8 < w/h < 1.2:
                clses[b, i, 0] = 1
    # ---------------------

    # 3. 缩放 Box 到评估分辨率 (513, 750)
    # 这是一个拷贝操作，避免污染原数据
    # 【修改点】：判断是 Tensor 还是 Numpy，使用对应的方法
    if hasattr(boxes, 'clone'):
        boxes_scaled = boxes.clone() # 如果是 Tensor
    else:
        boxes_scaled = boxes.copy()  # 如果是 Numpy Array
    boxes_scaled[:, :, ::2] *= EVAL_W
    boxes_scaled[:, :, 1::2] *= EVAL_H
    
    # 转换为 numpy 方便后续处理 (部分函数需要)
    clses_np = clses.cpu().numpy() if torch.is_tensor(clses) else clses
    boxes_np = boxes_scaled.cpu().numpy() if torch.is_tensor(boxes_scaled) else boxes_scaled
    
    # 4. 可视化 (可选)
    save_figs(names, clses_np, boxes_np, save_dir)
    
    # 5. 计算指标
    print("-" * 30)
    print("Calculating Metrics...")
    
    # Validity
    print("metrics_val (Validity):", metrics_val((EVAL_W, EVAL_H), clses_np, boxes_np))
    
    # Overlap (lower better)
    print("metrics_ove (Overlap):", metrics_ove(clses_np, boxes_np))
    
    # Alignment (lower better)
    print("metrics_ali (Alignment):", metrics_ali(clses_np, boxes_np))
    
    # Underlay Effectiveness (higher better)
    print("metrics_und_l (Underlay Large):", metrics_und_l(clses_np, boxes_np))
    print("metrics_und_s (Underlay Strict):", metrics_und_s(clses_np, boxes_np))
    
    # Saliency / Aesthetics based
    print("metrics_uti (Utility):", metrics_uti(names, clses_np, boxes_np))
    print("metrics_occ (Occlusion):", metrics_occ(names, clses_np, boxes_np))
    print("metrics_rea (Readability):", metrics_rea(names, clses_np, boxes_np))
    print("-" * 30)'''
     
if __name__ == "__main__":
    main()