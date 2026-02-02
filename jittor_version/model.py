#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
from jittor import nn
import os
# Jittor 内置了常用模型，替代 timm
from jittor.models import resnet50, resnet18 
# 确保目录下有 designSeq.py
from designSeq import reorder

# 手动实现的 Jittor LSTM 类，适配老版本 Jittor
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for i in range(num_layers):
            for j in range(self.num_directions):
                suffix = "_reverse" if j == 1 else ""
                layer_in = input_size if i == 0 else hidden_size * self.num_directions
                #【关键修改】：使用 setattr 动态定义参数名（如 weight_ih_l0），以确保手动实现的类能完美匹配 Jittor 权重文件中的 Key 路径
                setattr(self, f"weight_ih_l{i}{suffix}", jt.randn((hidden_size * 4, layer_in)))
                setattr(self, f"weight_hh_l{i}{suffix}", jt.randn((hidden_size * 4, hidden_size)))
                setattr(self, f"bias_ih_l{i}{suffix}", jt.zeros((hidden_size * 4,)))
                setattr(self, f"bias_hh_l{i}{suffix}", jt.zeros((hidden_size * 4,)))

    def cell_forward(self, x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        # 执行门控逻辑
        gates = jt.matmul(x, w_ih.t()) + b_ih + jt.matmul(h_prev, w_hh.t()) + b_hh
        i, f, g, o = jt.split(gates, self.hidden_size, dim=-1)
        i, f, o = jt.sigmoid(i), jt.sigmoid(f), jt.sigmoid(o)
        g = jt.tanh(g)
        c_curr = f * c_prev + i * g
        h_curr = o * jt.tanh(c_curr)
        return h_curr, c_curr

    def execute(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(1, 0, 2) # [Seq, Batch, Dim]
        
        seq_len, batch_size, _ = x.shape
        
        # 初始化隐藏状态
        if hx is None:
            h_states = jt.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
            c_states = jt.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
        else:
            h_states, c_states = hx

        current_input = x
        for l in range(self.num_layers):
            layer_outputs = []
            
            # 正向计算
            h_f, c_f = h_states[l*self.num_directions], c_states[l*self.num_directions]
            w_ih_f, w_hh_f = getattr(self, f"weight_ih_l{l}"), getattr(self, f"weight_hh_l{l}")
            b_ih_f, b_hh_f = getattr(self, f"bias_ih_l{l}"), getattr(self, f"bias_hh_l{l}")
            
            fwd_hs = []
            for t in range(seq_len):
                h_f, c_f = self.cell_forward(current_input[t], h_f, c_f, w_ih_f, w_hh_f, b_ih_f, b_hh_f)
                fwd_hs.append(h_f)
            fwd_hs = jt.stack(fwd_hs, dim=0)

            if self.bidirectional:
                # 反向计算
                h_r, c_r = h_states[l*2+1], c_states[l*2+1]
                w_ih_r, w_hh_r = getattr(self, f"weight_ih_l{l}_reverse"), getattr(self, f"weight_hh_l{l}_reverse")
                b_ih_r, b_hh_r = getattr(self, f"bias_ih_l{l}_reverse"), getattr(self, f"bias_hh_l{l}_reverse")
                
                rev_hs = []
                for t in reversed(range(seq_len)):
                    h_r, c_r = self.cell_forward(current_input[t], h_r, c_r, w_ih_r, w_hh_r, b_ih_r, b_hh_r)
                    rev_hs.append(h_r)
                rev_hs = jt.stack(rev_hs[::-1], dim=0)
                current_input = jt.concat([fwd_hs, rev_hs], dim=-1)
            else:
                current_input = fwd_hs

        output = current_input
        if self.batch_first:
            output = output.transpose(1, 0, 2)
            
        return output, (h_states, c_states)

class ResnetBackbone(nn.Module):
    def __init__(self, args):
        super(ResnetBackbone, self).__init__()
        
        # --- 替换 timm，使用 jittor.models ---
        if args["backbone"] == "resnet50":
            resnet = resnet50(pretrained=False)
            weight_path = "model_weight/resnet50.jtp" # 建议使用 jittor 格式
            ch = [1024, 2048]
        else:
            resnet = resnet18(pretrained=False)
            weight_path = "model_weight/resnet18.jtp"
            ch = [256, 512]

        # 加载权重逻辑 (Jittor 加载方式)
        if os.path.exists(weight_path):
            print(f"Loading local weights from {weight_path}...")
            pretrained_dict = jt.load(weight_path)
            model_dict = resnet.state_dict()
            
            #【关键修改】：Jittor 加载权重前需手动进行 state_dict 比对与形状检查。
            # 因为Jittor 的 load_parameters 无法自动处理 Key 不匹配，手动过滤可防止因框架差异导致的 C++ 底层崩溃
            new_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            if len(new_pretrained_dict) == 0:
                print("Warning: No matching keys found in weights file!")
            else:
                print(f"Successfully matched {len(new_pretrained_dict)} / {len(model_dict)} keys.")
                resnet.load_parameters(new_pretrained_dict)
        else:
            print(f"Pretrained weights not found at {weight_path}, using random init.")

        # --- 适配 4 通道输入 ---
        # Jittor 的 Conv2d 参数与 PyTorch 基本一致
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 提取 ResNet 层 (Jittor 模型结构稍有不同，直接取对应 layer)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3  # tilconv4
        self.layer4 = resnet.layer4  # conv5
        
        ## FPN
        self.fpn_conv11_4 = nn.Conv2d(ch[0], 256, 1)
        self.fpn_conv11_5 = nn.Conv2d(ch[1], 256, 1)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, padding=1)
        self.proj = nn.Conv2d(512, 8 * args["max_elem"], 1)
        self.fc_h0 = nn.Linear(330, args["num_layers"] * 2)
        
    def execute(self, img): # forward -> execute
        # Multi-scale feature
        x = self.layer0(img)
        x = self.layer1(x)
        x = self.layer2(x)
        resnet_f4 = self.layer3(x)
        resnet_f5 = self.layer4(resnet_f4)
        
        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        
        # F.interpolate -> nn.interpolate
        resnet_f5up = nn.interpolate(resnet_f5p, size=(resnet_f4p.shape[2], resnet_f4p.shape[3]), mode="nearest")
        
        # torch.concat -> jt.concat
        resnet_fused = jt.concat([resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1)
        resnet_proj = self.proj(resnet_fused)
        
        # flatten 处理
        resnet_flat = resnet_proj.reshape(resnet_proj.shape[0], resnet_proj.shape[1], -1)
        
        h0 = self.fc_h0(resnet_flat).transpose(2, 0, 1)
        return h0

class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        # 拆分 Sequential
        self.conv1d = nn.Conv1d(args["in_channels"], args["out_channels"], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
            
        # 这里不再预定义 nn.Pool 对象，我们在 execute 里直接用函数式写法
        # 老版本jittor没有内置LSTM，使用自己实现的LSTM类
        self.lstm = LSTM(input_size=args["out_channels"], hidden_size=args["hidden_size"],
                            num_layers=args["num_layers"], batch_first=True, bidirectional=True)

    def execute(self, layout, h0):
        # Jittor 不需要 flatten_parameters()
        x = layout.reshape(layout.shape[0], layout.shape[1], -1).transpose(0, 2, 1)
        
        x = self.conv1d(x)
        x = self.relu(x)
        
        # --- 解决 Jittor 池化尺寸校验报错 ---
        # 1.【关键迁移逻辑】：通过复制宽度维（Width=2）绕过 Jittor 池化算子对输入尺寸必须严格大于卷积核的校验限制
        # 现在 x 的形状从 [B, C, N] 变成 [B, C, N, 2]
        x_wide = jt.concat([x.unsqueeze(3), x.unsqueeze(3)], dim=3)
        
        # 2. 调用池化。由于 Width 是 2，Kernel 是 1，满足 Size(2) > Kernel(1)
        # H 维度是 N(10)，Kernel 是 3，满足 Size(10) > Kernel(3)
        x_pooled = nn.pool(x_wide, kernel_size=(3, 1), stride=1, padding=(1, 0), op='max')
        
        # 3. 取出其中一列，还原回 [B, C, N]
        x = x_pooled[:, :, :, 0]
        # ----------------------------------------------
        
        x = x.transpose(0, 2, 1)
        
        #【关键修改】：Jittor 的 LSTM (及自定义版本) 始终返回 (output, (h, c)) 元组，必须显式解构以获取序列特征
        output, (h, n) = self.lstm(x, (jt.zeros_like(h0), h0))
        
        # 强制检查输出维度。
        # 如果 output 变成了 [N, B, H]，将其转回 [B, N, H]
        if output.shape[0] != x.shape[0]:
            output = output.transpose(1, 0, 2)
        return output

class generator(nn.Module):
    def __init__(self, args):
        super(generator, self).__init__()
        self.resnet_fpn = ResnetBackbone(args)
        self.cnnlstm = CNN_LSTM(args)
        self.fc1 = nn.Linear(2 * args["hidden_size"], args["output_size"] // 2)
        self.fc2 = nn.Linear(2 * args["hidden_size"], args["output_size"] // 2)
        
    def execute(self, img, layout):
        h0 = self.resnet_fpn(img)
        lstm_output = self.cnnlstm(layout, h0)
        # nn.Softmax -> nn.softmax
        cls = nn.softmax(self.fc1(lstm_output), dim=-1)
        box = jt.sigmoid(self.fc2(lstm_output))
        return cls, box

class discriminator(nn.Module):
    def __init__(self, args):
        super(discriminator, self).__init__()
        self.resnet_fpn = ResnetBackbone(args)
        self.argmax = ArgMax() # Jittor 中调用方式不变
        self.cnnlstm = CNN_LSTM(args)
        self.fc_tf = nn.Linear(2 * args["hidden_size"], 1)

    def execute(self, img, layout):
        h0 = self.resnet_fpn(img)
        # 调用自定义 Function
        processed_layout = self.argmax(layout)
        # 调用 cnnlstm
        lstm_output_all = self.cnnlstm(processed_layout, h0)
        # lstm_output_all 形状为 [Batch, Seq, Feature]
        #【关键修改】：判别器仅需序列最后的全局特征，通过 [:, -1, :] 切片确保拿到的是 [Batch, Hidden] 维度
        lstm_output = lstm_output_all[:, -1, :] 
        tf = self.fc_tf(lstm_output)
        tf = jt.tanh(tf)
        return tf

class ArgMax(jt.Function):
    def execute(self, x):
        # x shape: [B, N, 2, 4]
        cls_probs = x[:, :, 0, :]
        
        # 1. 类别处理
        # 加上 [0] 修复之前提到的 tuple 报错
        idx = jt.argmax(cls_probs, dim=-1)[0].unsqueeze(-1)
        output_cls = jt.zeros_like(cls_probs)
        output_cls = output_cls.scatter(-1, idx, jt.array(1.0))
        
        # 2. 框处理 (显式取出 Var，避免对原 x 进行复杂索引同步)
        box_part = x[:, :, 1, :]
        
        # 3.【关键修改】：在调用 .numpy() 前必须执行 jt.sync 进行全局同步，否则会导致 GPU 数据未就绪触发段错误(Segfault)
        jt.sync([output_cls, box_part])
        
        cls_np = output_cls.numpy()
        box_np = box_part.numpy()
        
        b_s, n_e = cls_np.shape[:2]
        # 使用副本进行操作
        new_box_np = box_np.copy()
        
        for i in range(b_s):
            # 处理 Padding
            for j in range(n_e):
                if cls_np[i, j, 3] == 1:
                    new_box_np[i, j, :] = 0
            
            # 这里的 labels_idx 必须从 cls_np 计算
            labels_idx = cls_np[i].argmax(-1)
            order = reorder(labels_idx, new_box_np[i], "cxcywh")
            
            orig_box_i = new_box_np[i].copy()
            for j in range(min(len(order), n_e)):
                new_box_np[i, j, :] = orig_box_i[int(order[j])]

        # 4. 封装回 Jittor 变量
        # 显式构造，避免原地修改
        res_cls_var = output_cls.unsqueeze(2)
        res_box_var = jt.array(new_box_np).unsqueeze(2)
        
        #【关键修改】：Jittor 变量具有不可变性，此处通过 concat 构造新 Var 返回，严禁在 jt.Function 内部对输入 x 进行原地修改
        res = jt.concat([res_cls_var, res_box_var], dim=2)
        #【关键修改】：Jittor 的 jt.sync 接口强制要求传入列表(list)或元组(tuple)，严禁直接传入单个 Var 对象
        jt.sync([res])
        return res

    def grad(self, grad_output):
        # GAN 训练中，经过 ArgMax 的分支梯度通常不回传，或者直传
        return grad_output

def box_xyxy_to_cxcywh(x):
    # Jittor 没有 unbind，使用切片，使用 ... 确保切片作用于最后一维
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return jt.stack(b, dim=-1)

def random_init(batch, max_elem):
    # torch 替换为 jt
    cls_1 = jt.randint(0, 3, shape=(batch, max_elem, 1))
    
    cls = jt.zeros((batch, max_elem, 4))
    
    #【关键修改】：Jittor 的 scatter 算子第三个参数必须是 jt.Var 类型（如 jt.array(1.0)），传入 Python 原生 float 会导致 shape 属性报错
    cls = cls.scatter(-1, cls_1, jt.array(1.0))
    
    # Jittor 的正态分布
    box_xyxy = jt.randn((batch, max_elem, 1, 4)) * 0.15 + 0.5
    box = box_xyxy_to_cxcywh(box_xyxy)
    
    return jt.concat([cls.unsqueeze(2), box], dim=2)