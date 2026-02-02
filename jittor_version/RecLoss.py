#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
from jittor import nn
import numpy as np
from scipy.optimize import linear_sum_assignment

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#【关键修改】：Jittor Var 不支持 unbind 算子，此处改用切片索引获取坐标，并使用 ... (省略号) 确保逻辑适配多维输入
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jt.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    #【关键修改】：Jittor 中执行张量元素级比较需使用 jt.maximum/minimum 替代 torch.max/min
    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = jt.clamp(rb - lt, min_v=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6), union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = jt.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = jt.clamp(rb - lt, min_v=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-6)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    def execute(self, outputs, targets):
        with jt.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # 1. 展平预测结果 [Batch * Queries, ...]
            out_prob = nn.softmax(outputs["pred_logits"].reshape(-1, outputs["pred_logits"].shape[-1]), dim=-1)
            out_bbox = outputs["pred_boxes"].reshape(-1, 4)

            # 2. 准备目标数据
            tgt_ids = jt.concat([v["labels"] for v in targets])
            tgt_bbox = jt.concat([v["boxes"] for v in targets])

            # 3. 计算全局代价矩阵 [Batch * Queries, Total_Targets_In_Batch]
            cost_class = -out_prob[:, tgt_ids]
            cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # 总代价 [B*Q, Total_T]
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            #【关键修改】：SciPy 的匈牙利匹配算法必须在 CPU 运行，调用 .numpy() 会显式触发数据从 GPU 到 CPU 的同步点
            C_np = C.numpy()

            # 4.【关键修改】：Jittor 环境下弃用 np.split，改为显式按 Batch 维护目标偏移量并手动切片，确保在各样本目标框数量不等时逻辑依然对齐
            indices = []
            sizes = [len(v["boxes"]) for v in targets]
            
            # tgt_idx_start 用于记录当前图片的目标在 C_np 列中的起始位置
            tgt_idx_start = 0
            for i in range(bs):
                # 图片 i 的预测行区间: [i*Q : (i+1)*Q]
                row_start = i * num_queries
                row_end = (i + 1) * num_queries
                # 图片 i 的目标列区间: [tgt_idx_start : tgt_idx_start + sizes[i]]
                col_start = tgt_idx_start
                col_end = tgt_idx_start + sizes[i]
                # 如果该图没有目标框，跳过
                if sizes[i] > 0:
                    # 提取该图片的子矩阵 [Q, T_i]
                    cost_matrix_i = C_np[row_start:row_end, col_start:col_end]
                    # 运行匈牙利算法
                    idx_q, idx_t = linear_sum_assignment(cost_matrix_i)
                    indices.append((jt.array(idx_q, dtype='int64'), jt.array(idx_t, dtype='int64')))
                else:
                    # 空目标处理
                    indices.append((jt.array([], dtype='int64'), jt.array([], dtype='int64')))
                
                tgt_idx_start += sizes[i]
                
            return indices

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, coef_list, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.empty_weight = jt.array(coef_list, dtype='float32')

    def loss_labels(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # 1.【关键修改】：Jittor 执行多维 Var 索引赋值（如 target_classes[idx_b, idx_q]）要求索引对象必须显式转换为 int64 类型
        # idx_b: 每个匹配项属于哪个 Batch
        # idx_q: 每个匹配项属于哪个 Query 槽位
        idx_b = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)]).cast('int64')
        idx_q = jt.concat([src for (src, _) in indices]).cast('int64')
        
        # 2. 提取对应的目标类别
        target_classes_o = jt.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 3. 创建全量标签矩阵，初始化为背景类 ID (self.num_classes)
        # target_classes 维度: [Batch, Queries]
        target_classes = jt.full(src_logits.shape[:2], self.num_classes, dtype='int64')
        
        # 4. 使用 Var 索引进行赋值
        if idx_b.shape[0] > 0:
            target_classes[idx_b, idx_q] = target_classes_o
        
        # 5.【关键修改】：Jittor 的 nn.cross_entropy_loss 在多维广播时极其严格，此处通过展平为 2D/1D 结构避开 Shape Mismatch 报错，且计算结果等价
        # 将 logits 从 [B, Q, C] 转为 [B*Q, C]
        logits_flat = src_logits.reshape(-1, src_logits.shape[-1])
        # 将 target 从 [B, Q] 转为 [B*Q]
        target_flat = target_classes.reshape(-1)
        
        # 此时权重 self.empty_weight (形状为 [Classes]) 会正确作用于 C 维度
        loss_ce = nn.cross_entropy_loss(
            logits_flat, 
            target_flat, 
            weight=self.empty_weight
        )
        
        return {'loss_ce': loss_ce * self.weight_dict['loss_ce']}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx_b = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        idx_q = jt.concat([src for (src, _) in indices])
        src_boxes = outputs['pred_boxes'][idx_b, idx_q]
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = (src_boxes - target_boxes).abs().sum() / num_boxes
        giou_matrix = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        #【关键修改】：使用 jt.diag 提取 Pair-wise 矩阵的对角线元素，实现预测框与对应目标框的 GIoU 损失计算
        loss_giou = (1 - jt.diag(giou_matrix)).sum() / num_boxes
        return {
            'loss_bbox': loss_bbox * self.weight_dict['loss_bbox'],
            'loss_giou': loss_giou * self.weight_dict['loss_giou']
        }

    def execute(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        #【关键修改】：将归一化因子显式转为 Python 数值并进行除零保护，防止 Jittor 在计算梯度回传时因 0 维张量产生异常逻辑
        num_boxes_val = max(float(num_boxes), 1.0)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes_val))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes_val))
        return losses