#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
from jittor import nn
# 确保 dataset.py 已迁移为继承 jt.dataset.Dataset
from datasets import canvas 
from model import generator
import os
import numpy as np
import traceback # 用于打印详细错误

# --- 配置区域 ---
TEST_BG_PATH = "data/inpainted_1x"   
TEST_SAL_DIR = "data/saliency_map"
TEST_JSON_PATH = "data/test.json"     
CKPT_PATH = "train_result/DS-GAN-Epoch300.jtp"  # 建议使用 .jtp 后缀
OUTPUT_DIR = "output"
# ----------------

# 设置随机种子
jt.seed(0)
np.random.seed(0)

# 开启 GPU
if jt.has_cuda:
    jt.flags.use_cuda = 1

def box_xyxy_to_cxcywh(x):
    #【关键修改】：使用 ... (省略号) 确保切片始终作用于最后一维坐标，以兼容 Jittor 中形状可能为 [B,N,4] 或 [B,N,1,4] 的多维张量
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    #【关键修改】：使用 ... (省略号) 确保切片始终作用于最后一维坐标，以兼容 Jittor 中形状可能为 [B,N,4] 或 [B,N,1,4] 的多维张量
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)
    
def random_init(batch, max_elem):
    # 适配 Jittor 的随机采样与 scatter
    coef = [0.1, 0.8, 1, 1] 
    prob = np.array(coef) / sum(coef)
    cls_1_np = np.random.choice(4, size=(batch, max_elem, 1), p=prob)
    cls_1 = jt.array(cls_1_np)
    
    cls = jt.zeros((batch, max_elem, 4))
    #【关键修改】：Jittor 的 scatter 算子第三个参数必须是 jt.Var 类型，传入 Python float 会报错
    cls = cls.scatter(-1, cls_1, jt.array(1.0))
    
    # Jittor 正态分布: jt.randn
    # 形状改为 (batch, max_elem, 4)
    box_xyxy = jt.randn((batch, max_elem, 4)) * 0.15 + 0.5
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    # 拼接维度对齐 [B, N, 2, 4]
    init_layout = jt.concat([cls.unsqueeze(2), box.unsqueeze(2)], dim=2)
    return init_layout

def test(G, testing_set, fix_noise):
    G.eval()
    clses = []
    boxes = []
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Starting inference...")
    
    #【关键修改】：使用 jt.no_grad() 禁用梯度计算，Jittor 会在推理阶段自动优化算子执行效率并节省显存
    with jt.no_grad():
        #【关键修改】：Jittor 弃用 DataLoader 包装类，直接通过迭代 Dataset 对象即可获取批处理（Batch）数据
        for i, imgs in enumerate(testing_set):
            
            # 动态调整 noise (Jittor 获取 Batch Size 使用 .shape[0])
            current_batch_size = imgs.shape[0]
            if current_batch_size != fix_noise.shape[0]:
                curr_noise = fix_noise[:current_batch_size]
            else:
                curr_noise = fix_noise
            
            # 模型推理
            cls, box = G(imgs, curr_noise)
            
            #【关键修改】：Jittor 的 jt.argmax(dim) 返回 (index, value) 元组，需通过 [0] 显式提取索引 Var 才能进行后续计算
            pred_cls = jt.argmax(cls, dim=-1)[0].unsqueeze(-1)
            #【关键修改】：Jittor 的 .numpy() 会自动完成计算图分离（detach）和 GPU 到 CPU 的同步，无需显式调用 .cpu()
            clses.append(pred_cls.numpy())
            
            # box 转换并转 numpy
            boxes.append(box_cxcywh_to_xyxy(box).numpy())
            
            if i % 10 == 0:
                print(f"Processed batch {i}")

    if len(clses) > 0:
        # 拼接并转换为 numpy
        clses_final = np.concatenate(clses, axis=0)
        boxes_final = np.concatenate(boxes, axis=0)
        
        #【关键修改】：使用 jt.save 保存预测结果，以确保结果文件能被 eval.py 脚本中的 jt.load 算子正确解析
        jt.save(clses_final, os.path.join(OUTPUT_DIR, "clses-Epoch300.pt"))
        jt.save(boxes_final, os.path.join(OUTPUT_DIR, "boxes-Epoch300.pt"))
        print(f"Results saved to {OUTPUT_DIR}")
    else:
        print("No data processed.")

def main():
    test_batch_size = 4
    max_elem = 10
    
    # --- 增加：绝对路径调试逻辑 ---
    print(f"DEBUG: Current Working Directory is {os.getcwd()}")
    print(f"DEBUG: Looking for JSON at {os.path.abspath(TEST_JSON_PATH)}")
    
    if not os.path.exists(TEST_JSON_PATH):
        print(f"FATAL: JSON file not found at {os.path.abspath(TEST_JSON_PATH)}")
        print("Please check if you mounted the data volume correctly.")
        return
    
    if not os.path.exists(TEST_BG_PATH):
        print(f"FATAL: Image directory not found at {os.path.abspath(TEST_BG_PATH)}")
        return
    # ---------------------------

    # 1. 加载数据集
    try:
        print("Attempting to initialize canvas dataset...")
        testing_set = canvas(TEST_BG_PATH, TEST_SAL_DIR, TEST_JSON_PATH)
        #【关键修改】：在 Jittor 中通过 set_attrs 方法直接在数据集对象上配置 BatchSize 和是否开启并行/打乱
        testing_set.set_attrs(batch_size=test_batch_size, shuffle=False)
        print(f"Testing set size: {len(testing_set)}")
        
        if hasattr(testing_set, 'img_list'):
            jt.save(testing_set.img_list, "test_order.pt")
            print(f"Saved test_order.pt with {len(testing_set.img_list)} names.")
        else:
            print("Error: testing_set has no attribute 'img_list'.")
            return

    except Exception as e:
        print(f"Error during dataset initialization!")
        import traceback
        traceback.print_exc() # 这行一定会打印出具体是哪一行报错
        return

    # 2. 模型初始化
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
        ckpt = jt.load(CKPT_PATH)
        
        # 重新映射：将 "cnnlstm.lstm.weight_ih_l0" 直接对应到 G 里的同名参数
        # 现在的手动 LSTM 类已经拥有了这些同名成员
        new_ckpt = {}
        for k, v in ckpt.items():
            # 去掉 DataParallel 的 module. 前缀（如果有的话）
            name = k.replace("module.", "")
            new_ckpt[name] = v
            
        #【关键修改】：使用 load_parameters 接口加载权重字典，Jittor 会根据字典中的键值对自动匹配模型参数
        G.load_parameters(new_ckpt)
        print("Generator loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    test(G, testing_set, fix_noise)

if __name__ == "__main__":
    main()