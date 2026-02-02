#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 确保 dataset.py 是我们之前修改过的版本
from dataset import canvasLayout, canvas 
import numpy as np
from model import generator, discriminator
# 确保目录下有 RecLoss.py 和 utils.py
from RecLoss import SetCriterion, HungarianMatcher
from utils import setup_seed
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import OrderedDict

# --- 【配置区域】请根据实际情况修改路径 ---
TRAIN_INP_DIR = "data/inpainted_1x"       # 训练集背景图
TRAIN_SAL_DIR = "data/saliency_map"       # 训练集显著图
TRAIN_JSON_PATH = "data/train.json"       # 训练集 JSON 标注

TEST_BG_DIR = "data/inpainted_1x"         # 测试集背景图
TEST_SAL_DIR = "data/saliency_map"        # 测试集显著图
TEST_JSON_PATH = "data/test.json"        # 测试集 JSON 标注
# ------------------------------------------

setup_seed(0)
gpu = torch.cuda.is_available()
# 自动检测可用 GPU 数量
device_ids = list(range(torch.cuda.device_count())) if gpu else []
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
    
def random_init(batch, max_elem):
    """
    生成随机初始布局噪声。
    修改：适配 3 个类别 (Text, Logo, Underlay)。
    """
    # 类别概率分布：Text(0), Logo(1), Underlay(2)。
    # 下面概率表示生成 Text 的概率较大
    coef = [0.7, 0.1, 0.2] #此处修改，原来0.4，0.3，0.3，理由：让初始分布更接近真实海报：大部分是文本，少量 Logo 和 Underlay！！！！！
    
    # 随机选择类别 0, 1, 2
    cls_indices = torch.tensor(np.random.choice(3, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef)))
    
    # 初始化 one-hot 向量，长度为 4 以保持与模型 input_channels=8 兼容 (4 box + 4 cls)
    # 即使我们只用前3类，保留4维结构更安全
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_indices, 1)
    
    # 随机生成 Box (中心点 0.5, 宽高 0.15)
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    # 拼接 [Batch, Max_Elem, 8]
    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)
    return init_layout
    

def train(G, D, training_dl, criterionRec, criterionAdv, w_criterionAdv, optimizerG, optimizerD, schedulerG, schedulerD, epoch_n, max_elem):
    for idx, (image, label) in enumerate(training_dl):
        b_s = image.size(0)
        image = image.to(device)
        label = label.to(device)
        
        all_real = torch.ones(b_s, dtype=torch.float).to(device)
        all_fake = torch.full((b_s,), -1, dtype=torch.float).to(device)
        
        init_layout = random_init(b_s, max_elem).to(device)
        
        G.train()
        D.train()
        
        # -----------------
        #  Train Generator
        # -----------------
        G.zero_grad()
        # Generator 前向传播
        cls, box = G(image, init_layout)
        
        # 拼接生成的类别和框，作为 Discriminator 的输入
        # 注意：label 的维度需要匹配，如果 G 输出 cls 是 [B, N, 4]，label 也是
        # 将类别和框拼在一起，形成[B,N,8]的Tensor
        label_f = torch.concat([cls.unsqueeze(2), box.unsqueeze(2)], dim=2)
        # 接受背景图和生成的布局
        outputG = D(image, label_f)
        D_G_z1 = outputG.mean()
        
        # 解析 Ground Truth 用于重构损失
        cls_gt = label[:, :, 0] # [Batch, Max_Elem, 4] (One-hot)
        box_gt = label[:, :, 1] # [Batch, Max_Elem, 4] (Coords)
        
        outputs = {
            "pred_logits": cls,
            "pred_boxes": box.float()
        }
        # 转换 target 格式以适配 HungarianMatcher
        targets = [{
            "labels": torch.argmax(c, dim=-1).long(), 
            "boxes": b.float()
        } for c, b in zip(cls_gt, box_gt)]
        lossG = criterionAdv(outputG.view(-1), all_real)
        # 计算重构损失 (BBox loss + Class loss)
        dict_loss = criterionRec(outputs, targets)
        lossRec = sum(dict_loss.values())
        
        lossesG = w_criterionAdv * lossG + lossRec
        lossesG.backward()
        optimizerG.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        D.zero_grad()
        # Fake data
        outputD_f = D(image, label_f.detach())
        lossD_f = criterionAdv(outputD_f.view(-1), all_fake)
        D_G_z2 = outputD_f.mean()
        
        # Real data
        outputD_r = D(image, label)
        lossD_r = criterionAdv(outputD_r.view(-1), all_real)
        D_x = outputD_r.mean()
        
        lossesD = w_criterionAdv * (lossD_r + lossD_f)
        lossesD.backward()
        optimizerD.step()
                
        if idx % 10 == 0:
            print('[Epoch %d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch_n, idx, len(training_dl), lossesD.item(), lossesG.item(), D_x.item(), D_G_z1.item(), D_G_z2.item()))
    
    schedulerD.step()
    schedulerG.step()
        

def test(G, testing_dl, epoch_n, img_path_list):
    global fix_init_layout
    G.eval()
    
    # 保存模型
    if epoch_n % 50 == 0 or epoch_n == 300: # 每50个epoch保存一次
        save_path = f"output/DS-GAN-Epoch{epoch_n}.pth"
        if len(device_ids) > 1:
             torch.save(G.module.state_dict(), save_path)
        else:
             torch.save(G.state_dict(), save_path)
        print(f"Model saved: {save_path}")

    # 可视化第一批次
    with torch.no_grad():
        # 获取一个 batch
        imgs = next(iter(testing_dl))
        imgs = imgs.to(device)
        
        # 处理 fix_noise 大小可能与 batch 不匹配的问题
        curr_batch = imgs.size(0)
        curr_noise = fix_init_layout[:curr_batch].to(device)
        
        cls, box = G(imgs, curr_noise)
        
        plt.figure(figsize=(12, 5))
        for idx in range(curr_batch):
            c = cls[idx].detach().cpu()
            b = box[idx].detach().cpu()
            
            # 处理预测结果
            c_idx = torch.argmax(c, dim=1) # 获取类别索引
            b_xyxy = box_cxcywh_to_xyxy(b)
            
            # 打印调试信息 (可选)
            # print(f"Img {idx}: Classes {c_idx.numpy()}")
            
            # 缩放 Box 到可视化尺寸 (513x750 是原论文标准，这里保持一致以便对比)
            b_xyxy[:, ::2] *= 513
            b_xyxy[:, 1::2] *= 750
            
            # 读取背景原图
            # img_path_list 是 dataset.bg，包含了完整路径
            try:
                pil_img = Image.open(img_path_list[idx]).convert("RGB").resize((513, 750))
                drawn = draw_box_f(pil_img, c_idx.numpy(), b_xyxy.numpy())
                
                plt.subplot(1, 4, idx+1)
                plt.axis("off")
                plt.imshow(drawn)
            except Exception as e:
                print(f"Error visualizing image {idx}: {e}")

        save_plot_path = f"output/training_plot/Epoch{epoch_n}.png"
        plt.savefig(save_plot_path)
        plt.close()
        print(f"Validation plot saved to {save_plot_path}")

def draw_box_f(img, cls_list, box_list):
    img_copy = img.copy()
    draw = ImageDraw.ImageDraw(img_copy)
    # 【修改点】: 颜色映射适配 PosterLLaVa 类别
    # 0: Text(Green), 1: Logo(Red), 2: Underlay(Orange)
    cls_color_dict = {0: 'green', 1: 'red', 2: 'orange', 3: 'gray'}
    
    for cls, box in zip(cls_list, box_list):
        # 过滤掉无效框 (例如宽高极小) 或 Padding 类 (如果 padding 设为 3)
        if (box[2]-box[0]) > 1 and (box[3]-box[1]) > 1:
            color = cls_color_dict.get(int(cls), 'white')
            try:
                draw.rectangle(list(box), fill=None, outline=color, width=3)
            except ValueError:
                pass # 坐标异常忽略
    return img_copy

def isbase(name):
    if name.startswith("module.resnet_fpn") or name.startswith("resnet_fpn"):
        return True
    return False
    
def main():
    global fix_init_layout
    
    # 参数配置
    train_batch_size = 32  # 如果显存不够，改小这个值 (如 16 或 8)
    test_batch_size = 4
    max_elem = 10          # 设置为 PosterLLaVa 适合的长度 (如 10)
    epoch = 300
    linear_step = 100
    
    # 创建输出目录
    os.makedirs("output/training_plot", exist_ok=True)
    
    # 1. 初始化数据集
    # 【关键修改】：调用方式改变，传入 JSON，移除 sal_dir_2
    print("Initializing Datasets...")
    try:
        training_set = canvasLayout(TRAIN_INP_DIR, TRAIN_SAL_DIR, TRAIN_JSON_PATH, max_elem)
        training_dl = DataLoader(training_set, num_workers=4, batch_size=train_batch_size, shuffle=True, drop_last=True)
        
        # 修改这里：传入 TEST_JSON_PATH
        testing_set = canvas(TEST_BG_DIR, TEST_SAL_DIR, json_path=TEST_JSON_PATH) 
        
        testing_dl = DataLoader(testing_set, num_workers=4, batch_size=test_batch_size, shuffle=False)
        print(f"Training samples: {len(training_set)}, Testing samples: {len(testing_set)}")
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    # 2. 定义模型参数
    args_g = {
        "backbone": "resnet50",
        "in_channels": 8,   # 4 Class + 4 Coord
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 4,
        "output_size": 8,
        "max_elem": max_elem
    }
    args_d = {
        "backbone": "resnet18",
        "in_channels": 8,
        "out_channels": 32,
        "hidden_size": max_elem * 8,
        "num_layers": 2,
        "output_size": 8,
        "max_elem": max_elem
    }
    
    G = generator(args_g)
    D = discriminator(args_d)
    
    if gpu:
        G = G.to(device)
        D = D.to(device)
        if len(device_ids) > 1:
            print(f"Using {len(device_ids)} GPUs")
            G = torch.nn.DataParallel(G, device_ids=device_ids)
            D = torch.nn.DataParallel(D, device_ids=device_ids)
    
    # 3. 定义损失函数和优化器
    criterionAdv = nn.HingeEmbeddingLoss()
    matcher = HungarianMatcher(2, 5, 2)
    weight_dict = {
        "loss_ce": 4, "loss_bbox": 5, "loss_giou": 5#此处有修改（原来是252），理由：原本的代码中通过面积和长宽比强制修正类别，这说明模型没能学好“形状”和“类别”的对应关系。你需要提高 GIoU Loss 的权重。GIoU Loss 直接关系到框的重叠程度和长宽比（Aspect Ratio），提高它能让模型更精准地预测出“大而扁”或“小而方”的形状。！！！！！
    }
    
    # 类别权重：如果有 class 3 (padding)，可以给它低权重
    # 这里 SetCriterion 的 num_classes 参数很重要
    # 我们有 3 个真实类别 (0,1,2)。通常传入 3。
    # 且 eos_coef (coef 的最后一个) 是针对 "no object" 的权重
    coef_list = [1.0, 10.0, 5.0, 0.5] # Text, Logo, Underlay, EOS，此处修改，原本全1.0，理由：PosterLLaVa 数据集中，Text 应该是最多的，Logo 最少。，提高Text值以提升正确性，提高Logo值避免不敢预测Logo！！！！！
    
    # SetCriterion(num_classes, ...)
    criterionRec = SetCriterion(3, matcher, weight_dict, coef_list, ['labels', 'boxes']).to(device)
    
    # 固定测试用的噪声
    fix_init_layout = random_init(test_batch_size, max_elem)
    
    # 分组参数优化 (区分 Backbone 和 Head 的学习率)
    paramsG = list(filter(lambda kv: not isbase(kv[0]), G.named_parameters()))
    base_paramsG = list(filter(lambda kv: isbase(kv[0]), G.named_parameters()))
    
    paramsD = list(filter(lambda kv: not isbase(kv[0]), D.named_parameters()))
    base_paramsD = list(filter(lambda kv: isbase(kv[0]), D.named_parameters()))
    
    # 提取 parameter 对象
    paramsG = [p for n, p in paramsG]
    base_paramsG = [p for n, p in base_paramsG]
    paramsD = [p for n, p in paramsD]
    base_paramsD = [p for n, p in base_paramsD]
    
    optimizerG = optim.Adam([
        {"params": paramsG, "lr": 1e-4},
        {"params": base_paramsG, "lr": 1e-5}
    ])
    optimizerD = optim.Adam([
        {"params": paramsD, "lr": 2e-4},#此处修改，原来1e-3，理由：降低 D 的学习率，使其与 G 接近。，防止因D学习过快，导致G无法学习（梯度消失）！！！！！
        {"params": base_paramsD, "lr": 2e-5}#此处修改，原来1e-4，同上！！！！！
    ])
    
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=list(range(0, epoch, 50)), gamma=0.8)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=list(range(0, epoch, 25)), gamma=0.8)
    
    print("Start Training...")
    
    # 4. 训练循环
    for e in range(1, epoch + 1):
        # 动态调整对抗损失权重 (Warm-up)
        if e > linear_step:
            w_L_adv = 0.01#此处有修改，原本是1，理由：将对抗损失的上限大幅降低，建议限制在 0.01 或 0.1，让模型主要基于 Ground Truth 学习，GAN 只是用来“润色”合理性。！！！！！
        else:
            w_L_adv = 0.01 / linear_step * (e - 1)#此处修改同上！！！！！
            
        train(G, D, training_dl, criterionRec, criterionAdv, w_L_adv, optimizerG, optimizerD, schedulerG, schedulerD, e, max_elem)
        
        # 测试与保存
        # 注意：传入 testing_set.bg (背景图文件路径列表)
        # dataset.py 中 canvas 类的 self.bg 必须是完整路径列表
        # 如果 dataset.py 的 canvas.bg 存储的是路径，直接传
        test(G, testing_dl, e, testing_set.bg)
        

if __name__ == "__main__":
    main()