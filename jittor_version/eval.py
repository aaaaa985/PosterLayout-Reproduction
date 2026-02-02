#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import jittor as jt
import copy
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import log
import matplotlib.pyplot as plt

# --- 配置区域 ---
EVAL_W = 513
EVAL_H = 750
EPOCH_LIST = [50, 100, 150, 200, 250, 300] 
PLOT_SAVE_PATH = "output/metrics_plot.png"
IMG_DIR = "data/inpainted_1x"   
SAL_DIR = "data/saliency_map"   

if jt.has_cuda:
    jt.flags.use_cuda = 1

# 1. 基础工具函数

def draw_box(img, elems, elems2):
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    cls_color_dict = {0: 'green', 1: 'red', 2: 'orange'}
    W, H = img.size
    
    for cls, box in elems:
        # 确保 cls 是整数
        c_idx = int(cls[0]) if hasattr(cls, '__len__') else int(cls)
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        if (box_w * box_h) / (W * H) > 0.9: continue
        if c_idx in cls_color_dict:
            draw_ol.rectangle(tuple(box), fill=None, outline=cls_color_dict[c_idx], width=5)
    
    s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        c_idx = int(cls[0]) if hasattr(cls, '__len__') else int(cls)
        if ((box[2]-box[0])*(box[3]-box[1])) / (W * H) > 0.9: continue
        if c_idx in cls_color_dict:
            draw_f.rectangle(tuple(box), fill=cls_color_dict[c_idx])
            
    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.3))
    return Image.alpha_composite(drawn_outline, drawn_fill)

def cvt_pilcv(img, req='pil2cv', color_code=None):
    if req == 'pil2cv':
        if color_code is None: color_code = cv2.COLOR_RGB2BGR
        return cv2.cvtColor(np.asarray(img), color_code)
    elif req == 'cv2pil':
        if color_code is None: color_code = cv2.COLOR_BGR2RGB
        return Image.fromarray(cv2.cvtColor(img, color_code))
    return img

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    if np.max(grad_xy) > 0: grad_xy = grad_xy / np.max(grad_xy) * 255
    return Image.fromarray(grad_xy).convert('L')

# 2. 指标核心函数

def metrics_iou(bb1, bb2):
    w_inter = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    h_inter = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if w_inter <= 0 or h_inter <= 0: return 0
    a_inter = w_inter * h_inter
    a_union = (bb1[2]-bb1[0])*(bb1[3]-bb1[1]) + (bb2[2]-bb2[0])*(bb2[3]-bb2[1]) - a_inter
    return a_inter / (a_union + 1e-6)

def metrics_inter_oneside(bb1, bb2):
    w_inter = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    h_inter = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if w_inter <= 0 or h_inter <= 0: return 0
    return (w_inter * h_inter) / ((bb2[2]-bb2[0])*(bb2[3]-bb2[1]) + 1e-6)

def metrics_val(img_size, clses, boxes):
    total, empty = 0, 0
    for cls, box in zip(clses, boxes):
        for b in box:
            if (b[2]-b[0]) > 1 and (b[3]-b[1]) > 1:
                total += 1
                xl, yl, xr, yr = max(0, b[0]), max(0, b[1]), min(EVAL_W, b[2]), min(EVAL_H, b[3])
                if abs((xr - xl) * (yr - yl)) < (EVAL_W*EVAL_H/1000): empty += 1
    return 1 - empty / total if total > 0 else 0

def metrics_uti(img_names, clses, boxes):
    metrics = 0
    for idx, name in enumerate(img_names):
        try:
            pic = np.array(Image.open(os.path.join(SAL_DIR, name)).convert("L").resize((EVAL_W, EVAL_H))) / 255
            c_pic, cal_mask = 1.0 - pic, np.zeros_like(pic)
            for mb in boxes[idx]:
                if (mb[2]-mb[0]) > 1 and (mb[3]-mb[1]) > 1:
                    xl, yl, xr, yr = max(0, int(mb[0])), max(0, int(mb[1])), min(EVAL_W, int(mb[2])), min(EVAL_H, int(mb[3]))
                    cal_mask[yl:yr, xl:xr] = 1
            if np.sum(c_pic) > 0: metrics += (np.sum(c_pic * cal_mask) / np.sum(c_pic))
        except: continue
    return metrics / len(img_names)

def metrics_rea(img_names, clses, boxes):
    metrics = 0
    for idx, name in enumerate(img_names):
        try:
            pic = Image.open(os.path.join(IMG_DIR, name)).convert("RGB").resize((EVAL_W, EVAL_H))
            img_g_xy = np.array(img_to_g_xy(pic)) / 255
            cal_mask = np.zeros_like(img_g_xy)
            for i, c in enumerate(clses[idx]):
                mb = boxes[idx][i]
                #【关键修改】：适配推理阶段保存的 [N, 1] 类别索引格式，显式提取标量值进行类型判定
                c_val = int(c[0]) if hasattr(c, '__len__') else int(c)
                xl, yl, xr, yr = max(0, int(mb[0])), max(0, int(mb[1])), min(EVAL_W, int(mb[2])), min(EVAL_H, int(mb[3]))
                if c_val == 0: cal_mask[yl:yr, xl:xr] = 1 # Text
                elif c_val == 2: cal_mask[yl:yr, xl:xr] = 0 # Underlay
            if np.sum(cal_mask) > 0: metrics += (np.sum(img_g_xy[cal_mask == 1]) / np.sum(cal_mask))
        except: continue
    return metrics / len(img_names)

#【关键修改】：适配推理阶段使用 jt.argmax(...)[0].unsqueeze(-1) 保存的 [N, 1] 类别索引格式，通过 int(val[0]) 显式提取标量值
def metrics_ove(clses, boxes):
    m = 0
    for c, b in zip(clses, boxes):
        #【关键修改】：从 Jittor 推理结果的 [N, 1] 形状中提取类别索引，用于过滤 Text 与 Logo 元素
        mask = [i for i, val in enumerate(c) if int(val[0]) in [0, 1]]
        mb = b[mask]
        if len(mb) > 1:
            ove = sum(metrics_iou(mb[i], mb[j]) for i in range(len(mb)) for j in range(i+1, len(mb)))
            m += ove / (len(mb)*(len(mb)-1)/2)
    return m / len(clses)

def metrics_und_l(clses, boxes):
    m, avali = 0, 0
    for c, b in zip(clses, boxes):
        mask_d = [i for i, val in enumerate(c) if int(val[0]) == 2]
        mask_o = [i for i, val in enumerate(c) if int(val[0]) in [0, 1]]
        if len(mask_d) > 0 and len(mask_o) > 0:
            avali += 1
            und = sum(max(metrics_inter_oneside(b[i], b[j]) for j in mask_o) for i in mask_d)
            m += und / len(mask_d)
    return m / avali if avali > 0 else 0

def metrics_und_s(clses, boxes):
    m, avali = 0, 0
    for c, b in zip(clses, boxes):
        mask_d = [i for i, val in enumerate(c) if int(val[0]) == 2]
        mask_o = [i for i, val in enumerate(c) if int(val[0]) in [0, 1]]
        if len(mask_d) > 0 and len(mask_o) > 0:
            avali += 1
            und = 0
            for i in mask_d:
                for j in mask_o:
                    if b[i][0]<=b[j][0] and b[i][1]<=b[j][1] and b[i][2]>=b[j][2] and b[i][3]>=b[j][3]:
                        und += 1; break
            m += und / len(mask_d)
    return m / avali if avali > 0 else 0

def metrics_ali(clses, boxes):
    m = 0
    for c, b in zip(clses, boxes):
        mask = [i for i, val in enumerate(c) if int(val[0]) in [0, 1]]
        mb = b[mask]
        if len(mb) <= 1: continue
        theda = np.array([[x[0]/EVAL_W, x[1]/EVAL_H, (x[0]+x[2])/(2*EVAL_W), (x[1]+x[3])/(2*EVAL_H), x[2]/EVAL_W, x[3]/EVAL_H] for x in mb])
        # 计算每一列的最小 delta
        g_vals = []
        for j in range(6):
            col = theda[:, j]
            min_delta = 1e9
            for row_i in range(len(col)):
                for row_k in range(row_i+1, len(col)):
                    min_delta = min(min_delta, abs(col[row_i] - col[row_k]))
            #【关键修改】：在对数计算中增加微小偏移量（1e-7），防止 Jittor 预测的极值坐标在计算 1 - min_delta 时因浮点数精度误差导致 log(0) 或负数报错
            g_vals.append(-log(1 - min_delta + 1e-7, 10))
        m += min(g_vals)
    return m / len(clses)

def metrics_occ(img_names, clses, boxes):
    metrics = 0
    for idx, name in enumerate(img_names):
        try:
            pic = np.array(Image.open(os.path.join(SAL_DIR, name)).convert("L").resize((EVAL_W, EVAL_H))) / 255
            mask = np.zeros_like(pic)
            for mb in boxes[idx]:
                if (mb[2]-mb[0]) > 1 and (mb[3]-mb[1]) > 1:
                    xl, yl, xr, yr = max(0, int(mb[0])), max(0, int(mb[1])), min(EVAL_W, int(mb[2])), min(EVAL_H, int(mb[3]))
                    mask[yl:yr, xl:xr] = 1
            if np.sum(mask) > 0: metrics += (np.sum(pic[mask == 1]) / np.sum(mask))
        except: continue
    return metrics / len(img_names)

def save_figs(names, clses, boxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for idx, name in enumerate(names):
        try:
            pic = Image.open(os.path.join(IMG_DIR, name)).convert("RGB").resize((EVAL_W, EVAL_H))
            # 这里的 zip 处理：c 和 b 都是 list
            drawn = draw_box(pic, zip(clses[idx], boxes[idx]), zip(clses[idx], boxes[idx]))
            #【关键修改】：Jittor 保存的文件名列表可能包含路径信息，使用 os.path.basename 清理路径，防止在保存可视化结果时因找不到子目录而崩溃
            drawn.save(os.path.join(save_dir, os.path.splitext(os.path.basename(name))[0] + "_vis.png"))
        except Exception as e: 
            print(f"Viz error for {name}: {e}")

def plot_metrics(epochs, history, save_path):
    plt.figure(figsize=(15, 8))
    for i, (name, values) in enumerate(history.items()):
        plt.subplot(3, 3, i + 1)
        plt.plot(epochs, values, marker='o', color='blue', linewidth=2)
        plt.title(name)
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Metrics plot saved to: {save_path}")

# 3. 主程序

def main():
    # 1. 加载测试顺序
    try:
        names_data = jt.load("test_order.pt")
        #【关键修改】：Jittor 的 jt.load 会根据保存内容自动识别。加载非 Var 对象（如文件名列表 list）时，加载结果可能被封装或保持原样，此处需做兼容转换处理
        if hasattr(names_data, 'numpy'):
            names = names_data.numpy().tolist()
        else:
            names = names_data
        print(f"Test samples: {len(names)}")
    except Exception as e:
        print(f"Error: test_order.pt load failed: {e}")
        return
    
    history = {k: [] for k in ['Validity', 'Overlap', 'Alignment', 'Underlay_L', 'Underlay_S', 'Utility', 'Occlusion', 'Readability']}
    valid_epochs = []

    for epoch in EPOCH_LIST:
        cp = f"output/clses-Epoch{epoch}.pt"
        bp = f"output/boxes-Epoch{epoch}.pt"
        
        if not (os.path.exists(cp) and os.path.exists(bp)):
            print(f"Epoch {epoch} files missing, skipping.")
            continue
            
        try:
            clses_data = jt.load(cp)
            boxes_data = jt.load(bp)
            valid_epochs.append(epoch)
            
            #【关键修改】：Jittor 的 .numpy() 直接实现显存到内存的同步。由于推理脚本保存格式的特殊性，此处增加 hasattr 判断以确保无论加载结果是 Var 还是 Numpy 数组均能安全运行
            clses_np = clses_data.numpy() if hasattr(clses_data, 'numpy') else clses_data
            boxes_np = boxes_data.numpy() if hasattr(boxes_data, 'numpy') else boxes_data
            
            # 缩放坐标到评估分辨率 [0, 1] -> [513, 750]
            #【关键修改】：Jittor 转出的 Numpy 数组共享内存。此处必须使用 .copy() 创建副本，防止在计算一个指标时修改原始坐标数据，导致后续指标重复缩放产生错误
            b_scaled = boxes_np.copy()
            b_scaled[:, :, 0::2] *= EVAL_W
            b_scaled[:, :, 1::2] *= EVAL_H
            
            print(f"\n--- Processing Epoch {epoch} ---")
            
            # 计算各项指标
            v = metrics_val((EVAL_W, EVAL_H), clses_np, b_scaled)
            o = metrics_ove(clses_np, b_scaled)
            a = metrics_ali(clses_np, b_scaled)
            ul = metrics_und_l(clses_np, b_scaled)
            us = metrics_und_s(clses_np, b_scaled)
            ut = metrics_uti(names, clses_np, b_scaled)
            occ = metrics_occ(names, clses_np, b_scaled)
            rea = metrics_rea(names, clses_np, b_scaled)

            # 存储历史
            history['Validity'].append(v); history['Overlap'].append(o)
            history['Alignment'].append(a); history['Underlay_L'].append(ul)
            history['Underlay_S'].append(us); history['Utility'].append(ut)
            history['Occlusion'].append(occ); history['Readability'].append(rea)

            # 实时打印
            for k in history: print(f"{k}: {history[k][-1]:.4f}")
            
            # 仅在最后一个 Epoch 进行可视化
            if epoch == EPOCH_LIST[-1]:
                print(f"Saving visualizations for Epoch {epoch}...")
                save_figs(names, clses_np, b_scaled, f"output/vis_epoch{epoch}")
                
        except Exception as e:
            print(f"Error @ Epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()

    if valid_epochs:
        plot_metrics(valid_epochs, history, PLOT_SAVE_PATH)
    else:
        print("No valid epochs found to plot.")

if __name__ == "__main__":
    main()