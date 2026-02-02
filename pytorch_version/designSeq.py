import numpy as np
import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    # 避免除零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.array(inter) / (np.array(union) + 1e-6)
    return iou

def reorder(cls, box, o="xyxy", max_elem=None):
    """
    对设计元素进行重排序
    """
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
    
    # 适配 PosterLLaVa / 新 dataset.py 的映射:
    # 0: Text, 1: Logo, 2: Underlay
    text = np.where(cls == 0)[0]
    logo = np.where(cls == 1)[0]
    deco = np.where(cls == 2)[0]
    
    # --- 关键修改：不要将 index 转为 numpy array 再转回 list，这会导致 int 变成 float ---
    # 我们直接在 list 上操作，或者在取值时强制转 int
    
    # 构造 (index, area) 列表
    area_list = area.tolist() if torch.is_tensor(area) else area
    indexed_area = list(enumerate(area_list))
    
    # 筛选并排序
    # 这里的 x[0] 是索引(int)，x[1] 是面积(float)
    text_items = [item for item in indexed_area if item[0] in text]
    deco_items = [item for item in indexed_area if item[0] in deco]
    
    order_text = sorted(text_items, key=lambda x: x[1], reverse=True)
    order_deco = sorted(deco_items, key=lambda x: x[1])
    
    # 建立连接关系
    connection = {}
    reverse_connection = {}
    
    for idx, _ in order_deco:
        idx = int(idx) # 确保是整数
        con = []
        
        # 检查 Logo 是否在 Underlay 上
        for idx_ in logo:
            idx_ = int(idx_)
            if iou[idx, idx_] > 0: 
                connection[idx_] = idx
                con.append(idx_)
                
        # 检查 Text 是否在 Underlay 上
        for idx_ in text:
            idx_ = int(idx_)
            if iou[idx, idx_] > 0:
                connection[idx_] = idx
                con.append(idx_)
                
        # 检查 Underlay 之间的重叠
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
                    
    # 开始构建排序列表
    
    # 1. 先处理 Logo
    for idx in logo:
        idx = int(idx)
        if idx in connection:
            d = connection[idx]
            if isinstance(d, list): d = d[0]
            d = int(d) # 确保 d 也是整数
            
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
                
    # 2. 再处理 Text
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
            
    # 3. 处理剩余的 Deco
    for idx in deco:
        idx = int(idx)
        if idx not in order:
            order.append(idx)

    # 最终转换，确保全是 Python int 类型，防止后续序列化报错
    return [int(x) for x in order][:min(len(cls), max_elem)]