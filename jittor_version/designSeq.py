#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import jittor as jt

def box_area(boxes):
    #【关键修改】：Jittor 核心库不包含 torchvision.ops.boxes，需手动实现 box_area 等基础几何算子以进行后续重排序计算
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_cxcywh_to_xyxy(x):
    #【关键修改】：Jittor 不支持 unbind，此处使用省略号 ... 配大切片索引最后一维，确保逻辑适配多维输入并规避切片溢出报错
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    """
    迁移说明: 
    1. torch.max/min 替换为 jt.maximum/jt.minimum (元素级最大/最小)
    2. clamp(min=0) 替换为 jt.clamp(min_v=0)
    """
    #【关键修改】：将输入显式转换为 jt.array，确保后续执行 jt.maximum 等算子时能正常触发 JIT 编译加速
    boxes1 = jt.array(boxes1)
    boxes2 = jt.array(boxes2)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 计算交集坐标: [N,M,2]
    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  

    # 计算交集宽高
    wh = jt.clamp(rb - lt, min_v=0)  
    inter = wh[:, :, 0] * wh[:, :, 1]  

    union = area1[:, None] + area2 - inter
    
    # 避免除零错误，并转为 numpy 供 reorder 逻辑对比
    iou = inter / (union + 1e-6)
    #【关键修改】：计算结束后显式调用 .numpy()，因为后续 reorder 逻辑包含大量 CPU 密集型循环判断
    return iou.numpy()

def reorder(cls, box, o="xyxy", max_elem=None):
    """
    对设计元素进行重排序
    """
    #【关键修改】：使用 jt.array 显式将输入（可能是 Numpy 或 List）转换为 Jittor 变量，确保后续所有几何计算能正常触发框架的算子加速
    box = jt.array(box)
    
    if o == "cxcywh":
        box = box_cxcywh_to_xyxy(box)
    if max_elem == None:
        max_elem = len(cls)
    
    # init
    order = []
    
    # convert
    cls = np.array(cls)
    area = box_area(box)
    
    # 计算所有框之间的 IoU
    iou = box_iou(box, box)
    
    # 适配 PosterLLaVa: 0: Text, 1: Logo, 2: Underlay
    text = np.where(cls == 0)[0]
    logo = np.where(cls == 1)[0]
    deco = np.where(cls == 2)[0]
    
    # 构造 (index, area) 列表
    #【关键修改】：将 torch.is_tensor 替换为 Jittor 原生的 jt.is_var 接口，用于判定对象是否为框架管理的动态变量
    area_list = area.tolist() if jt.is_var(area) else area
    indexed_area = list(enumerate(area_list))
    
    # 筛选并排序
    text_items = [item for item in indexed_area if item[0] in text]
    deco_items = [item for item in indexed_area if item[0] in deco]
    
    order_text = sorted(text_items, key=lambda x: x[1], reverse=True)
    order_deco = sorted(deco_items, key=lambda x: x[1])
    
    # 建立连接关系 (逻辑部分基本保持 numpy/python 原样)
    connection = {}
    reverse_connection = {}
    
    for idx, _ in order_deco:
        idx = int(idx)
        con = []
        
        for idx_ in logo:
            idx_ = int(idx_)
            if iou[idx, idx_] > 0: 
                connection[idx_] = idx
                con.append(idx_)
                
        for idx_ in text:
            idx_ = int(idx_)
            if iou[idx, idx_] > 0:
                connection[idx_] = idx
                con.append(idx_)
                
        for idx_ in deco:
            idx_ = int(idx_)
            if idx == idx_: continue
            if iou[idx, idx_] > 0:
                if idx_ not in connection:
                    connection[idx_] = [idx]
                else:
                    if isinstance(connection[idx_], list):
                        connection[idx_].append(idx)
                    else:
                        connection[idx_] = [connection[idx_], idx]
                con.append(idx_)
        reverse_connection[idx] = con
                    
    # 构建排序列表 (逻辑不变)
    for idx in logo:
        idx = int(idx)
        if idx in connection:
            d = connection[idx]
            if isinstance(d, list): d = d[0]
            d = int(d)
            d_group = reverse_connection.get(d, [])
            for idx_ in d_group:
                idx_ = int(idx_)
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            if idx not in order:
                order.append(idx)
                
    for idx, _ in order_text:
        idx = int(idx)
        if len(order) >= max_elem:
            break
        if idx in order: continue 
        if idx in connection:
            d = connection[idx]
            if isinstance(d, list): d = d[0]
            d = int(d)
            d_group = reverse_connection.get(d, [])
            for idx_ in d_group:
                idx_ = int(idx_)
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
            
    for idx in deco:
        idx = int(idx)
        if idx not in order:
            order.append(idx)

    return [int(x) for x in order][:min(len(cls), max_elem)]