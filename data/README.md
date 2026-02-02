# Data Directory Structure / 数据目录结构

---

## English

This directory contains the dataset used for **PosterLayout** reproduction, based on the **PosterLLaVa** dataset. 

**Note:** Due to storage limits, the actual image files are not uploaded to this repository. Please download the dataset manually and organize it as follows.

### Directory Tree

```text
data/
├── inpainted_1x/       # Background images (inpainted)
├── original_poster/    # Original poster images
├── saliency_map/       # Saliency maps (grayscale)
├── train.json          # Training annotations
└── test.json           # Testing annotations
```

### Data Description

- **inpainted_1x/**: Contains background images where original text and elements have been removed. These serve as the canvas for layout generation.
- **original_poster/**: The ground truth poster images before inpainting.
- **saliency_map/**: Grayscale images representing the visual importance of different regions in the background.
- **train.json / test.json**: JSON files containing element types, bounding boxes, and image dimensions.

### How to Get the Data

1. Download the **PosterLLaVa** dataset from the official source or the [PosterLLaVa](https://github.com/posterllava/PosterLLaVA) repository.
2. Place the folders and JSON files into this data/ directory according to the structure above.

------

## 中文说明

本目录存放基于 **PosterLLaVa** 数据集的海报布局生成数据。

**注意：** 由于文件体积限制，实际的图像文件未上传至 GitHub 仓库。请手动下载数据集并按以下结构存放。

### 目录树

```
data/
├── inpainted_1x/       # 背景底图（已修补，去除文字元素）
├── original_poster/    # 原始海报图
├── saliency_map/       # 显著性图（灰度图）
├── train.json          # 训练集标注文件
└── test.json           # 测试集标注文件
```

### 数据内容详述

- **inpainted_1x/**: 移除了原始文字和设计元素的底图，作为布局生成的输入背景。
- **original_poster/**: 原始的海报图像，用于参考或评估。
- **saliency_map/**: 显著性图，反映背景中各区域的视觉吸引力，辅助模型避开重要视觉中心。
- **train.json / test.json**: 包含海报中各元素的类别、坐标（Bbox）以及画布尺寸等信息。

### 获取方式

1. 从官方渠道或 [PosterLLaVa](https://github.com/posterllava/PosterLLaVA) 仓库下载 **PosterLLaVa** 数据集。
2. 将下载好的文件夹和 JSON 文件按照上述结构放入此 data/ 目录下即可运行代码。