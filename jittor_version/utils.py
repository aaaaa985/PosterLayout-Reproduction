#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jittor as jt
import numpy as np
import random

def setup_seed(seed):
    #【关键修改】：使用 jittor 统一的 jt.seed 接口替代 PyTorch 繁琐的各种设备种子设置
    jt.seed(seed)
    
    # 保持 numpy 和 python 原生随机库的种子设置
    np.random.seed(seed)
    random.seed(seed)
    
    #【关键修改】：移除 PyTorch 特有的 cudnn.deterministic，Jittor 的 JIT 机制会自动保证算子确定性