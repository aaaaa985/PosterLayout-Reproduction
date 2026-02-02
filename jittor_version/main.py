#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
from jittor import nn, optim
# 假设 dataset.py 已迁移为继承 jt.dataset.Dataset
from datasets import canvasLayout, canvas 
import numpy as np
from model import generator, discriminator
# 假设 RecLoss.py 和 utils.py 已按照 Jittor 规范修改
from RecLoss import SetCriterion, HungarianMatcher
from utils import setup_seed
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# --- 【配置区域】 ---
TRAIN_INP_DIR = "data/inpainted_1x"
TRAIN_SAL_DIR = "data/saliency_map"
TRAIN_JSON_PATH = "data/train.json"

TEST_BG_DIR = "data/inpainted_1x"
TEST_SAL_DIR = "data/saliency_map"
TEST_JSON_PATH = "data/test.json"
# --------------------

# 设置全局随机种子
setup_seed(0)

# Jittor 自动管理 GPU，开启以下标志即可使用 GPU
if jt.has_cuda:
    jt.flags.use_cuda = 1

def box_xyxy_to_cxcywh(x):
    #【【关键修改】：使用 ... (省略号) 确保切片始终作用于最后一维坐标，以适配 Jittor [B,N,4] 或 [B,N,1,4] 的多维输出
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    #同上
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)
    
def random_init(batch, max_elem):
    # 生成随机初始布局噪声 (Jittor 版)
    coef = [0.7, 0.1, 0.2]
    # 使用 numpy 生成分布，再转回 jittor
    cls_indices_np = np.random.choice(3, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef))
    cls_indices = jt.array(cls_indices_np)
    
    cls = jt.zeros((batch, max_elem, 4))
    cls = cls.scatter(-1, cls_indices, 1.0)
    
    # Jittor 正态分布生成方式
    box_xyxy = jt.randn((batch, max_elem, 1, 4)) * 0.15 + 0.5
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    return jt.concat([cls.unsqueeze(2), box], dim=2)

def train(G, D, training_loader, criterionRec, criterionAdv, w_criterionAdv, optimizerG, optimizerD, schedulerG, schedulerD, epoch_n, max_elem):
    G.train()
    D.train()
    
    # Jittor 数据集直接迭代
    for idx, (image, label) in enumerate(training_loader):
        b_s = image.shape[0]
        
        #【关键修改】：Jittor 使用字符串声明数据类型（如 'float32'），而非 torch.float32 对象
        all_real = jt.ones((b_s,), dtype='float32')
        all_fake = jt.full((b_s,), -1.0, dtype='float32')
        
        init_layout = random_init(b_s, max_elem)
        
        # -----------------
        #  Train Generator
        # -----------------
        cls, box = G(image, init_layout)
        label_f = jt.concat([cls.unsqueeze(2), box.unsqueeze(2)], dim=2)
        
        outputG = D(image, label_f)
        D_G_z1 = outputG.mean()
        
        # 解析 Ground Truth
        cls_gt = label[:, :, 0, :] 
        box_gt = label[:, :, 1, :]
        
        outputs = {
            "pred_logits": cls,
            "pred_boxes": box
        }
        targets = [{
            #【关键修改】：Jittor 的 jt.argmax(dim) 返回 (index, value) 元组，需通过 [0] 提取索引以匹配 targets 格式
            "labels": jt.argmax(c, dim=-1)[0].long(), 
            "boxes": b
        } for c, b in zip(cls_gt, box_gt)]
        
        lossG_adv = criterionAdv(outputG.view(-1), all_real)
        dict_loss = criterionRec(outputs, targets)
        lossRec = sum(dict_loss.values())
        
        lossesG = w_criterionAdv * lossG_adv + lossRec
        
        # Jittor 推荐 optimizer.step(loss) 自动处理 backward 和 zero_grad
        optimizerG.step(lossesG)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Fake
        outputD_f = D(image, label_f.detach())
        lossD_f = criterionAdv(outputD_f.view(-1), all_fake)
        D_G_z2 = outputD_f.mean()
        
        # Real
        outputD_r = D(image, label)
        lossD_r = criterionAdv(outputD_r.view(-1), all_real)
        D_x = outputD_r.mean()
        
        lossesD = w_criterionAdv * (lossD_r + lossD_f)
        optimizerD.step(lossesD)
                
        if idx % 10 == 0:
            print('[Epoch %d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch_n, idx, len(training_loader), lossesD.item(), lossesG.item(), D_x.item(), D_G_z1.item(), D_G_z2.item()))
    
    schedulerD.step()
    schedulerG.step()

def test(G, testing_loader, epoch_n, img_path_list, fix_init_layout):
    G.eval()
    
    # Jittor 保存权重
    if epoch_n % 50 == 0 or epoch_n == 300:
        save_path = f"output/DS-GAN-Epoch{epoch_n}.jtp"
        G.save(save_path)
        print(f"Model saved: {save_path}")

    # 可视化
    with jt.no_grad():
        # 获取第一个 batch
        imgs, _ = next(iter(testing_loader))
        curr_batch = imgs.shape[0]
        curr_noise = fix_init_layout[:curr_batch]
        
        cls, box = G(imgs, curr_noise)
        
        plt.figure(figsize=(12, 5))
        for idx in range(min(curr_batch, 4)):
            c = cls[idx]
            b = box[idx]
            #【关键修改】：Jittor 的 .numpy() 会自动触发显存到内存的同步，无需像 PyTorch 那样显式调用 .detach().cpu()
            #【关键修改】：Jittor 的 jt.argmax 在带 dim 参数时返回 (index, value) 元组，必须通过 [0] 索引取回索引 Var 才能进行后续计算
            c_idx = jt.argmax(c, dim=1)[0].numpy()
            b_xyxy = box_cxcywh_to_xyxy(b.unsqueeze(0))[0].numpy()
            
            # 缩放
            b_xyxy[:, ::2] *= 513
            b_xyxy[:, 1::2] *= 750
            
            try:
                pil_img = Image.open(img_path_list[idx]).convert("RGB").resize((513, 750))
                drawn = draw_box_f(pil_img, c_idx, b_xyxy)
                
                plt.subplot(1, 4, idx+1)
                plt.axis("off")
                plt.imshow(drawn)
            except Exception as e:
                print(f"Viz Error: {e}")

        os.makedirs("output/training_plot", exist_ok=True)
        plt.savefig(f"output/training_plot/Epoch{epoch_n}.png")
        plt.close()

def draw_box_f(img, cls_list, box_list):
    # 绘图部分基本保持 PIL 逻辑，Jittor 转 numpy 即可
    img_copy = img.copy()
    draw = ImageDraw.ImageDraw(img_copy)
    cls_color_dict = {0: 'green', 1: 'red', 2: 'orange', 3: 'gray'}
    
    for cls, box in zip(cls_list, box_list):
        if (box[2]-box[0]) > 1 and (box[3]-box[1]) > 1:
            color = cls_color_dict.get(int(cls), 'white')
            draw.rectangle(list(box), fill=None, outline=color, width=3)
    return img_copy

def isbase(name):
    if "resnet_fpn" in name:
        return True
    return False
    
def main():
    # 参数
    train_batch_size = 32
    test_batch_size = 4
    max_elem = 10
    epoch = 300
    linear_step = 100
    
    # Jittor 数据集初始化
    print("Initializing Datasets...")
    training_set = canvasLayout(TRAIN_INP_DIR, TRAIN_SAL_DIR, TRAIN_JSON_PATH, max_elem)
    # Jittor 设置批处理和随机打乱
    training_set.set_attrs(batch_size=train_batch_size, shuffle=True)
    
    testing_set = canvas(TEST_BG_DIR, TEST_SAL_DIR, json_path=TEST_JSON_PATH)
    testing_set.set_attrs(batch_size=test_batch_size, shuffle=False)

    # 模型加载
    args_g = {"backbone": "resnet50", "in_channels": 8, "out_channels": 32, "hidden_size": max_elem*8, "num_layers": 4, "output_size": 8, "max_elem": max_elem}
    args_d = {"backbone": "resnet18", "in_channels": 8, "out_channels": 32, "hidden_size": max_elem*8, "num_layers": 2, "output_size": 8, "max_elem": max_elem}
    
    G = generator(args_g)
    D = discriminator(args_d)
    
    # 损失函数
    criterionAdv = nn.HingeEmbeddingLoss()
    matcher = HungarianMatcher(2, 5, 2)
    weight_dict = {"loss_ce": 4, "loss_bbox": 5, "loss_giou": 5}
    coef_list = [1.0, 10.0, 5.0, 0.5]
    
    criterionRec = SetCriterion(3, matcher, weight_dict, coef_list, ['labels', 'boxes'])
    
    fix_init_layout = random_init(test_batch_size, max_elem)
    
    # Jittor 优化器分层设置
    paramsG = []
    base_paramsG = []
    for n, p in G.parameters_dict().items():
        if isbase(n): base_paramsG.append(p)
        else: paramsG.append(p)

    #【关键修改】：Jittor 的 Adam 构造函数要求即使在参数组中指定了 lr，也必须显式传入全局 lr 参数
    optimizerG = optim.Adam([
        {"params": paramsG, "lr": 1e-4},
        {"params": base_paramsG, "lr": 1e-5}
    ])
    
    paramsD = []
    base_paramsD = []
    for n, p in D.parameters_dict().items():
        if isbase(n): base_paramsD.append(p)
        else: paramsD.append(p)

    #【关键修改】：Jittor 的 Adam 构造函数要求即使在参数组中指定了 lr，也必须显式传入全局 lr 参数
    optimizerD = optim.Adam([
        {"params": paramsD, "lr": 2e-4},
        {"params": base_paramsD, "lr": 2e-5}
    ])
    
    schedulerG = jt.lr_scheduler.MultiStepLR(optimizerG, milestones=list(range(0, epoch, 50)), gamma=0.8)
    schedulerD = jt.lr_scheduler.MultiStepLR(optimizerD, milestones=list(range(0, epoch, 25)), gamma=0.8)
    
    for e in range(1, epoch + 1):
        if e > linear_step:
            w_L_adv = 0.01
        else:
            w_L_adv = 0.01 / linear_step * (e - 1)
            
        train(G, D, training_set, criterionRec, criterionAdv, w_L_adv, optimizerG, optimizerD, schedulerG, schedulerD, e, max_elem)
        test(G, testing_set, e, testing_set.bg, fix_init_layout)
        
if __name__ == "__main__":
    main()