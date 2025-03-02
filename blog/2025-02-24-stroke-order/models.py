import torch.nn as nn
import torch
import torchvision.models as models

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
    

class MobileNetV3Model(nn.Module):
    """MobileNetV3 model for stroke count prediction using torchvision implementation"""
    
    def __init__(self, num_strokes, num_classes):
        super().__init__()
        
        # 使用 torchvision 的 MobileNetV3-Small 预训练模型
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # 修改第一层以接受单通道输入
        original_first_layer = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 如果使用预训练权重，将三通道权重平均到单通道
        mobilenet.features[0][0].weight.data = original_first_layer.weight.data.sum(
            dim=1, keepdim=True
        ) / 3.0
        
        # 移除分类器
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        
        # 获取特征维度
        self.feature_dim = 576  # MobileNetV3-Small 的特征维度
        
        # 笔画数预测头
        self.stroke_count_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        
        # 第一个笔画预测头
        self.first_stroke_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # 预测笔画数和第一个笔画
        stroke_count = self.stroke_count_head(features)
        first_stroke_logits = self.first_stroke_head(features)
        
        return {
            'stroke_count': stroke_count.squeeze(-1),
            'first_stroke': first_stroke_logits
        }


class MobileNetV3LiteModel(nn.Module):
    """Lightweight MobileNetV3 model without pretrained weights"""
    
    def __init__(self, num_strokes, num_classes):
        super().__init__()
        
        # 使用 torchvision 的 MobileNetV3-Small，不使用预训练权重
        mobilenet = models.mobilenet_v3_small(pretrained=False)
        
        # 修改第一层以接受单通道输入
        mobilenet.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # 可选：减少网络深度以进一步减轻过拟合
        # 这里我们保留前8个块（约一半的网络）
        reduced_features = nn.Sequential(*list(mobilenet.features.children())[:8])
        self.features = reduced_features
        
        # 添加全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 获取特征维度 - 需要根据截断的网络计算
        # MobileNetV3-Small 在第8个块后的特征维度是48 (not 96)
        self.feature_dim = 48
        
        # 笔画数预测头 - 改为回归模型
        self.stroke_count_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 输出单个值作为笔画数预测
        )
        
        # 第一个笔画预测头
        self.first_stroke_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # 预测笔画数和第一个笔画
        stroke_count = self.stroke_count_head(features).squeeze(-1)  # 移除最后一个维度
        first_stroke_logits = self.first_stroke_head(features)
        
        return {
            'stroke_count': stroke_count,  # 现在是回归值而不是分类logits
            'first_stroke': first_stroke_logits
        }