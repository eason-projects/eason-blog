---
slug: stroke-order
title: 强化学习中文笔画顺序预测
authors: eason
tags: [ml, rl]
draft: true
---

我们看到随着具身智能（机器人）的蓬勃发展，
越来越多的机械臂应用面世。

在这些里面，就有通过驱动机器人来书写汉字的应用。

为了探索此类的应用，我们简化任务，我们希望通过输入图片来预测图片中汉字的笔画顺序。
以作为未来通过机器人书写汉字的基础。

<!-- truncate -->

## 提示词

```
我们希望构建一个RL的算法，通过输入汉字的图片，算法需要正确的输出汉字的笔画（笔顺）。

如何构建？选用什么样的方式实现并训练呢？

注：
1. 我们只要求预测笔画和顺序，不要求预测笔画的起始和结束位置。（不需要预测笔画的区域或者位置）
2. 推荐使用Stable-Baselines3。
3. 详细解释下状态（state）的设计。
4. 在训练的过程中，我们可以知道每一个汉字的笔画顺序。但是在使用模型结果的时候，我们不知道画面中的汉字的正确笔画，我们希望完全依赖模型给我们结果。
```

## Setup

```bash
# Cretae a venv
python3 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt
```

## 训练一个CNN网络模型

我们为了先得到一个处理文字图片的视觉模型，我们会先单独训练一个CNN的视觉网络模型。

为了和后续的笔画预测模型衔接，我们的模型输入选择64x64的图片，输出有两个：

1. 笔画个数预测
2. 第一个笔画预测

我们构造的网络结构如下：

```python

class CNNModel(nn.Module):
    """CNN model for stroke count prediction"""
    
    def __init__(self, num_strokes, num_classes):
        super().__init__()
        
         # CNN for processing images - same architecture as in CustomCombinedExtractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),  # Add dropout after pooling
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),  # Add dropout after pooling
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),  # Add dropout after pooling
            nn.Flatten()
        )
        
        # Update CNN output size (8x8x128 = 8192)
        self.cnn_output_size = 8192
        
        # Prediction heads
        self.stroke_count_head = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(256, num_strokes + 1)  # +1 for 0 strokes
        )
        
        self.first_stroke_head = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Process image
        features = self.cnn(x)
        
        # Predict stroke count and first stroke
        stroke_count_logits = self.stroke_count_head(features)
        first_stroke_logits = self.first_stroke_head(features)
        
        return {
            'stroke_count': stroke_count_logits,
            'first_stroke': first_stroke_logits
        }
```

我们在训练了70轮后，我们的第一个笔画的预测，接近100%，但是笔画个数预测，大约在50%左右徘徊。
因此我们需要调整策略。

### 数据增强

我们添加了一些增强的策略，其内容如下：

```python
self.augmentation_types = [
    "original",           # 0: No augmentation
    "slight_rotation",    # 1: Slight rotation
    "slight_translation", # 2: Slight translation
    "slight_scaling",     # 3: Slight scaling
    "slight_shear",       # 4: Slight shear
    "slight_blur",        # 5: Slight blur
    "elastic_deform",     # 6: Elastic deformation
    "noise",              # 7: Add noise
    "erosion",            # 8: Erosion - thins strokes
    "dilation"            # 9: Dilation - thickens strokes
]
```

但是我们训练后，其笔画数的预测，依然在40%左右徘徊。





## AI提供的建议

1. https://www.wenxiaobai.com/share/chat/f16629de-6eab-4efd-8b4f-b368f7a77e36
2. http://www.moe.gov.cn/jyb_sjzl/ziliao/A19/202103/W020210318300204215237.pdf
3. 