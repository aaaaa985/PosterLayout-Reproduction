#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
# 确保目录下有 designSeq.py
from designSeq import reorder

class ResnetBackbone(nn.Module):
    def __init__(self, args):
        # CNN backbone
        super(ResnetBackbone, self).__init__()
        
        # --- 修改点 1: 加载本地权重文件 ---
        # 请确保你的根目录下有 model_weight 文件夹，并且包含对应的 pth 文件
        if args["backbone"] == "resnet50":
            resnet = timm.create_model("resnet50", pretrained=False)
            weight_path = "model_weight/resnet50_a1_0-14fe96d1.pth"
            ch = [1024, 2048]
        else:
            resnet = timm.create_model("resnet18", pretrained=False)
            weight_path = "model_weight/resnet18-5c106cde.pth"
            ch = [256, 512]

        # 尝试加载本地权重
        if os.path.exists(weight_path):
            print(f"Loading local backbone weights from {weight_path}...")
            try:
                resnet.load_state_dict(torch.load(weight_path))
            except Exception as e:
                print(f"Error loading local weights: {e}. Trying to download...")
                resnet = timm.create_model(args["backbone"], pretrained=True)
        else:
            print(f"Warning: {weight_path} not found. Downloading pretrained weights...")
            resnet = timm.create_model(args["backbone"], pretrained=True)
            
        # 修改第一层卷积以接受 4 通道输入 (RGB + Saliency)
        # 注意：这会重置第一层的权重为随机初始化，这是符合原论文逻辑的
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        res_chi = list(resnet.children())
        self.resnet_tilconv4 = nn.Sequential(*res_chi[:7])
        self.resnet_conv5 = res_chi[7]
        
        ## FPN
        self.fpn_conv11_4 = nn.Conv2d(ch[0], 256, 1, 1, 0)
        self.fpn_conv11_5 = nn.Conv2d(ch[1], 256, 1, 1, 0)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.proj = nn.Conv2d(512, 8 * args["max_elem"], 1, 1, 0)
        self.fc_h0 = nn.Linear(330, args["num_layers"] * 2)
        
    def forward(self, img):
        # Multi-sacle feature
        resnet_f4 = self.resnet_tilconv4(img)
        resnet_f5 = self.resnet_conv5(resnet_f4)
        
        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        # 上采样融合
        resnet_f5up = F.interpolate(resnet_f5p, size=resnet_f4p.shape[2:], mode="nearest")
        resnet_fused = torch.concat([resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1)
        resnet_proj = self.proj(resnet_fused)
        resnet_flat = resnet_proj.flatten(start_dim=-2)
        
        h0 = self.fc_h0(resnet_flat).permute(2, 0, 1)
        return h0

class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args["in_channels"], out_channels=args["out_channels"], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=1, padding=1)
        )
        self.lstm = nn.LSTM(input_size=args["out_channels"], hidden_size=args["hidden_size"],
                            num_layers=args["num_layers"], batch_first=True, bidirectional=True)

    def forward(self, layout, h0):
        self.lstm.flatten_parameters()
        x = layout.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 1).contiguous()
        output, _ = self.lstm(x, (torch.zeros_like(h0).contiguous(), h0.contiguous()))
        
        return output

class generator(nn.Module):
    def __init__(self, args):
        super(generator, self).__init__()
        # 骨干网络：提取图像特征 (ResNet50 + FPN)
        self.resnet_fpn = ResnetBackbone(args)
        # 序列生成：CNN-LSTM 捕捉元素间的空间依赖
        self.cnnlstm = CNN_LSTM(args)
        # 预测头：分别预测类别(cls)和位置(box)
        self.fc1 = nn.Linear(2 * args["hidden_size"], args["output_size"] // 2)
        self.fc2 = nn.Linear(2 * args["hidden_size"], args["output_size"] // 2)
        
    def forward(self, img, layout):
        # Multi-sacle feature
        h0 = self.resnet_fpn(img)# 获取上下文特征
        lstm_output = self.cnnlstm(layout, h0)#根据上下文细化布局
        cls = nn.Softmax(dim=-1)(self.fc1(lstm_output))
        box = nn.Sigmoid()(self.fc2(lstm_output))
        return cls, box

class discriminator(nn.Module):
    def __init__(self, args):
        super(discriminator, self).__init__()
        
        # CNN backbone
        self.resnet_fpn = ResnetBackbone(args)

        # Differential argmax
        self.argmax = ArgMax()
        
        # CNN-LSTM
        self.cnnlstm = CNN_LSTM(args)
        
        # Predictor
        self.fc_tf = nn.Linear(2 * args["hidden_size"], 1)

    def forward(self, img, layout):
        h0 = self.resnet_fpn(img)
        lstm_output = self.cnnlstm(self.argmax.apply(layout), h0)[:, -1, :]
        tf = self.fc_tf(lstm_output)
        tf = nn.Tanh()(tf)
        return tf

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        idx = torch.argmax(x[:, :, 0], -1).unsqueeze(-1)
        output = torch.zeros_like(x[:, :, 0])
        output.scatter_(-1, idx, 1)
        x[:, :, 0] = output
        
        i_, j_ = x.shape[:2]
        for i in range(i_):
            for j in range(j_):
                # 【修改点 2】: 适配类别。我们约定 class 3 是 Padding/Invalid
                # 0:Text, 1:Logo, 2:Underlay, 3:Padding
                if x[i][j][0][3] == 1:
                    x[i][j][1] = torch.zeros_like(x[i][j][1])
                    
        for i in range(i_):
            order = reorder(x[i, :, 0].detach().cpu(), x[i, :, 1].detach().cpu(), "cxcywh")
            tmp = x[i, :, 1].clone()
            for j in range(j_):
                if j < len(order):
                    x[i][j][1] = tmp[int(order[j])]
                
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def random_init(batch, max_elem):
    # 【修改点 3】: 随机生成有效类别 (0, 1, 2)，避开 Padding (3)
    cls_1 = torch.randint(0, 3, size=(batch, max_elem, 1))
    
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    return torch.concat([cls.unsqueeze(2), box], dim=2)