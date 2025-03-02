import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os
import random
import mlflow
from tqdm import tqdm
import copy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from datetime import datetime
import torchvision.models as models
from collections import Counter

# Import the dataset class and model from pretrain_cnn.py
from pretrain_cnn import StrokeOrderPretrainDataset

# 添加数据增强类
class AugmentedStrokeOrderDataset(Dataset):
    """Dataset with augmentation for pretraining the CNN on stroke count prediction"""
    
    def __init__(self, base_dataset, augmentation_factor=10):
        """
        Args:
            base_dataset: The original StrokeOrderPretrainDataset
            augmentation_factor: Number of augmented versions to create for each sample
        """
        self.base_dataset = base_dataset
        self.augmentation_factor = augmentation_factor
        
        # Define augmentation types
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
            "dilation",           # 9: Dilation - thickens strokes
            "combined_1",         # 10: Combined augmentation 1 (rotation + blur)
            "combined_2"          # 11: Combined augmentation 2 (scaling + noise)
        ]
        
        # Ensure we don't exceed the number of available augmentations
        self.augmentation_factor = min(self.augmentation_factor, len(self.augmentation_types))
    
    def __len__(self):
        return len(self.base_dataset) * self.augmentation_factor
    
    def __getitem__(self, idx):
        # Determine which sample and which augmentation to use
        base_idx = idx // self.augmentation_factor
        aug_idx = idx % self.augmentation_factor
        
        # Get the base sample
        base_sample = self.base_dataset[base_idx]
        
        # If it's the original (aug_idx=0), return the base sample directly
        if aug_idx == 0:
            return base_sample
        
        # For augmentations, convert tensor to PIL image
        # The image tensor is in [0,1] range with shape [1, 64, 64]
        image_tensor = base_sample['image']
        
        # Convert to numpy array in [0,255] range
        image_np = (image_tensor.numpy().squeeze()).astype(np.uint8)
        
        # Create PIL image
        image_pil = Image.fromarray(image_np)
        
        # Apply augmentation based on aug_idx
        aug_type = self.augmentation_types[aug_idx]
        
        if aug_type == "slight_rotation":
            # Rotation ±15 degrees
            angle = random.uniform(-15, 15)
            augmented_pil = image_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        
        elif aug_type == "slight_translation":
            # Translation up to 15% in each direction
            width, height = image_pil.size
            dx = int(random.uniform(-0.15, 0.15) * width)
            dy = int(random.uniform(-0.15, 0.15) * height)
            augmented_pil = ImageOps.expand(image_pil, border=(0, 0, 0, 0), fill=255)
            augmented_pil = augmented_pil.transform(
                (width, height),
                Image.AFFINE,
                (1, 0, dx, 0, 1, dy),
                resample=Image.BILINEAR,
                fillcolor=255
            )
        
        elif aug_type == "slight_scaling":
            # Scaling 85-115%
            width, height = image_pil.size
            scale = random.uniform(0.85, 1.15)
            new_width = int(width * scale)
            new_height = int(height * scale)
            augmented_pil = image_pil.resize((new_width, new_height), Image.BILINEAR)
            
            # Center the resized image
            result = Image.new(image_pil.mode, (width, height), 255)
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            result.paste(augmented_pil, (paste_x, paste_y))
            augmented_pil = result
        
        elif aug_type == "slight_shear":
            # Slight shear
            width, height = image_pil.size
            shear_factor = random.uniform(-0.2, 0.2)
            augmented_pil = image_pil.transform(
                (width, height),
                Image.AFFINE,
                (1, shear_factor, 0, 0, 1, 0),
                resample=Image.BILINEAR,
                fillcolor=255
            )
        
        elif aug_type == "slight_blur":
            # Slight Gaussian blur
            augmented_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        
        elif aug_type == "elastic_deform":
            # Elastic deformation - simulates natural handwriting variations
            # Convert to binary image
            threshold = 128
            binary_np = np.array(image_pil) < threshold
            
            # Parameters for elastic deformation
            alpha = random.uniform(8, 12)  # Intensity of deformation
            sigma = random.uniform(3, 6)   # Smoothness of deformation
            
            # Create random displacement fields
            shape = binary_np.shape
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # Displace mesh grid
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Map coordinates
            distorted = map_coordinates(binary_np, indices, order=1).reshape(shape)
            
            # Convert back to uint8
            augmented_np = np.where(distorted, 0, 255).astype(np.uint8)
            augmented_pil = Image.fromarray(augmented_np)
        
        elif aug_type == "noise":
            # Add salt and pepper noise
            augmented_np = np.array(image_pil)
            
            # Add random black dots (pepper)
            pepper_prob = random.uniform(0.001, 0.015)
            pepper_mask = np.random.random(augmented_np.shape) < pepper_prob
            augmented_np[pepper_mask] = 0
            
            # Add random white dots (salt) - less noticeable on white background
            salt_prob = random.uniform(0.001, 0.008)
            salt_mask = np.random.random(augmented_np.shape) < salt_prob
            augmented_np[salt_mask] = 255
            
            augmented_pil = Image.fromarray(augmented_np)
        
        elif aug_type == "erosion":
            # Erosion - thins the strokes
            augmented_pil = image_pil.filter(ImageFilter.MinFilter(3))
        
        elif aug_type == "dilation":
            # Dilation - thickens the strokes
            augmented_pil = image_pil.filter(ImageFilter.MaxFilter(3))
            
        elif aug_type == "combined_1":
            # Combined augmentation 1: rotation + blur
            # First apply rotation
            angle = random.uniform(-10, 10)
            temp_pil = image_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
            
            # Then apply blur
            augmented_pil = temp_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
            
        elif aug_type == "combined_2":
            # Combined augmentation 2: scaling + noise
            # First apply scaling
            width, height = image_pil.size
            scale = random.uniform(0.9, 1.1)
            new_width = int(width * scale)
            new_height = int(height * scale)
            temp_pil = image_pil.resize((new_width, new_height), Image.BILINEAR)
            
            # Center the resized image
            result = Image.new(image_pil.mode, (width, height), 255)
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            result.paste(temp_pil, (paste_x, paste_y))
            temp_pil = result
            
            # Then add noise
            augmented_np = np.array(temp_pil)
            
            # Add random black dots (pepper)
            pepper_prob = random.uniform(0.001, 0.01)
            pepper_mask = np.random.random(augmented_np.shape) < pepper_prob
            augmented_np[pepper_mask] = 0
            
            augmented_pil = Image.fromarray(augmented_np)
        
        else:
            # Fallback to original
            augmented_pil = image_pil
        
        # Convert back to tensor in the same format as the original
        # First convert to numpy array in [0,1] range
        augmented_np = np.array(augmented_pil) / 255.0
        
        # Then convert to tensor with shape [1, 64, 64]
        augmented_tensor = torch.from_numpy(augmented_np).float().unsqueeze(0)
        
        # Create augmented sample
        augmented_sample = {
            'image': augmented_tensor,
            'stroke_count': base_sample['stroke_count'],
            'first_stroke': base_sample['first_stroke'],
            'character': base_sample['character']
        }
        
        return augmented_sample

def visualize_augmentations(dataset, num_samples=5):
    """Visualize augmentations for a few samples"""
    
    # Select random samples
    indices = random.sample(range(len(dataset.base_dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, dataset.augmentation_factor, figsize=(20, 3 * num_samples))
    
    for i, base_idx in enumerate(indices):
        # Get original sample
        base_sample = dataset.base_dataset[base_idx]
        char = base_sample['character']
        stroke_count = base_sample['stroke_count']
        
        # Display original and augmented versions
        for j in range(dataset.augmentation_factor):
            # Calculate index in augmented dataset
            aug_idx = base_idx * dataset.augmentation_factor + j
            
            # Get augmented sample
            aug_sample = dataset[aug_idx]
            
            # Display image
            ax = axes[i, j]
            
            # Convert tensor to numpy for display
            if j == 0:
                # Original image
                img = base_sample['image'].numpy().squeeze()
                title = f"{char} (Original)\nStrokes: {stroke_count}"
            else:
                # Augmented image - should maintain original black-on-white pattern
                img = aug_sample['image'].squeeze().numpy()
                title = f"{char} ({dataset.augmentation_types[j]})"
            
            # Use grayscale colormap for all images
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    print("Augmentation examples saved to 'augmentation_examples.png'")
    plt.show()

# 定义改进的CNN模型
class ImprovedCNNModel(nn.Module):
    """Improved CNN model for stroke count prediction with ResNet backbone"""
    
    def __init__(self, num_strokes, num_classes):
        super().__init__()
        
        # 使用预训练的ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=True)
        
        # 修改第一层以接受单通道输入
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # ResNet18的特征维度是512
        self.feature_dim = 512
        
        # 笔画数预测头
        self.stroke_count_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_strokes + 1)  # +1 for 0 strokes
        )
        
        # 第一个笔画预测头
        self.first_stroke_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 提取特征
        x = self.backbone(x)
        features = torch.flatten(x, 1)
        
        # 预测笔画数和第一个笔画
        stroke_count_logits = self.stroke_count_head(features)
        first_stroke_logits = self.first_stroke_head(features)
        
        return {
            'stroke_count': stroke_count_logits,
            'first_stroke': first_stroke_logits
        }

# 定义加权交叉熵损失函数
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits, targets, weight=self.weight)

# 定义焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 修改数据集类以支持类别平衡采样
class BalancedStrokeOrderDataset(Dataset):
    """Dataset with balanced sampling for stroke count classes"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        # 按笔画数对样本进行分组
        self.samples_by_stroke_count = {}
        for idx, sample in enumerate(base_dataset.samples):
            stroke_count = sample['stroke_count']
            if stroke_count not in self.samples_by_stroke_count:
                self.samples_by_stroke_count[stroke_count] = []
            self.samples_by_stroke_count[stroke_count].append(idx)
        
        # 计算每个笔画数类别的样本数量
        self.stroke_count_distribution = {k: len(v) for k, v in self.samples_by_stroke_count.items()}
        print("笔画数分布:", self.stroke_count_distribution)
        
        # 计算最大样本数
        self.max_samples = max(self.stroke_count_distribution.values())
        
        # 为每个类别创建采样索引
        self.indices = []
        for stroke_count, samples in self.samples_by_stroke_count.items():
            # 对小类进行过采样
            if len(samples) < self.max_samples:
                # 随机采样以达到最大样本数
                self.indices.extend(np.random.choice(samples, self.max_samples, replace=True))
            else:
                # 对大类直接使用所有样本
                self.indices.extend(samples)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始数据集中的样本
        orig_idx = self.indices[idx]
        return self.base_dataset[orig_idx]

# 修改训练函数
def train_model_with_augmentation(max_chars=7000, augmentation_factor=10, batch_size=32, num_epochs=50, learning_rate=0.001, use_balanced_dataset=True, use_focal_loss=True):
    """Train the CNN model with data augmentation and improved strategies"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:8090")
    
    # Set experiment name
    experiment_name = "stroke_order_cnn_improved"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        # Check if MPS is available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device for training")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU")
        
        # Create base datasets
        train_base_dataset = StrokeOrderPretrainDataset(
            stroke_order_path='./stroke-order-jian.json',
            stroke_table_path='./stroke-table.json',
            image_folder='./images',
            max_chars=max_chars,
            train=True,
            split_ratio=0.9
        )
        
        val_dataset = StrokeOrderPretrainDataset(
            stroke_order_path='./stroke-order-jian.json',
            stroke_table_path='./stroke-table.json',
            image_folder='./images',
            max_chars=max_chars,
            train=False,
            split_ratio=0.9
        )
        
        # 分析训练集中的笔画数分布
        stroke_counts = [sample['stroke_count'] for sample in train_base_dataset.samples]
        stroke_count_counter = Counter(stroke_counts)
        print("训练集笔画数分布:", dict(sorted(stroke_count_counter.items())))
        
        # 计算笔画数类别权重（反比于样本数量）
        total_samples = len(stroke_counts)
        class_weights = {}
        for stroke_count, count in stroke_count_counter.items():
            class_weights[stroke_count] = total_samples / (count * len(stroke_count_counter))
        
        # 转换为张量
        max_stroke_count = max(stroke_counts)
        weight_tensor = torch.ones(max_stroke_count + 1)
        for stroke_count, weight in class_weights.items():
            weight_tensor[stroke_count] = weight
        
        print("笔画数类别权重:", weight_tensor)
        
        # 使用平衡数据集或增强数据集
        if use_balanced_dataset:
            print("使用平衡采样数据集")
            train_dataset = BalancedStrokeOrderDataset(train_base_dataset)
        else:
            print("使用标准增强数据集")
            # Create augmented training dataset
            train_dataset = AugmentedStrokeOrderDataset(
                base_dataset=train_base_dataset,
                augmentation_factor=augmentation_factor
            )
        
        print(f"Base train dataset size: {len(train_base_dataset)}")
        print(f"Final train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # 如果使用增强数据集，可视化增强效果
        if not use_balanced_dataset:
            visualize_augmentations(train_dataset, num_samples=5)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Get number of stroke types and max stroke count
        max_stroke_count = max(sample['stroke_count'] for sample in train_base_dataset.samples)
        num_stroke_types = len(train_base_dataset.dataset.stroke_names)
        
        print(f"Max stroke count: {max_stroke_count}")
        print(f"Number of stroke types: {num_stroke_types}")
        
        # Create improved model
        model = ImprovedCNNModel(max_stroke_count, num_stroke_types)
        
        # 将权重张量移动到设备
        weight_tensor = weight_tensor.to(device)
        
        # Define loss function and optimizer
        if use_focal_loss:
            print("使用焦点损失函数")
            criterion = {
                'stroke_count': FocalLoss(gamma=2, weight=weight_tensor),
                'first_stroke': nn.CrossEntropyLoss()
            }
        else:
            print("使用加权交叉熵损失函数")
            criterion = {
                'stroke_count': WeightedCrossEntropyLoss(weight=weight_tensor),
                'first_stroke': nn.CrossEntropyLoss()
            }
        
        # 使用AdamW优化器，它有更好的权重衰减实现
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 使用余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=learning_rate/100
        )
        
        model.to(device)
        
        # Log parameters to MLflow
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_stroke_count": max_stroke_count,
            "num_stroke_types": num_stroke_types,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "augmentation_factor": augmentation_factor if not use_balanced_dataset else "N/A",
            "use_balanced_dataset": use_balanced_dataset,
            "use_focal_loss": use_focal_loss,
            "device": str(device),
            "model_type": "ImprovedCNNModel"
        })
        
        # Best model state
        best_val_loss = float('inf')
        best_model_state = None
        best_stroke_count_acc = 0.0
        
        # Early stopping parameters
        patience = 20
        counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_stroke_count_correct = 0
            train_first_stroke_correct = 0
            train_samples = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                # Move data to device
                images = batch['image'].to(device)
                stroke_counts = batch['stroke_count'].to(device)
                first_strokes = batch['first_stroke'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss_stroke_count = criterion['stroke_count'](outputs['stroke_count'], stroke_counts)
                loss_first_stroke = criterion['first_stroke'](outputs['first_stroke'], first_strokes)
                
                # 增加笔画数预测的权重
                loss = 2.0 * loss_stroke_count + loss_first_stroke
                
                # Backward pass and optimize
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
                train_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
                
                _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
                train_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
                
                train_samples += images.size(0)
            
            # Calculate epoch statistics
            train_loss = train_loss / train_samples
            train_stroke_count_acc = train_stroke_count_correct / train_samples
            train_first_stroke_acc = train_first_stroke_correct / train_samples
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_stroke_count_correct = 0
            val_first_stroke_correct = 0
            val_samples = 0
            
            # 用于分析每个笔画数类别的准确率
            stroke_count_correct_by_class = {}
            stroke_count_total_by_class = {}
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    # Move data to device
                    images = batch['image'].to(device)
                    stroke_counts = batch['stroke_count'].to(device)
                    first_strokes = batch['first_stroke'].to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    
                    # Calculate loss
                    loss_stroke_count = criterion['stroke_count'](outputs['stroke_count'], stroke_counts)
                    loss_first_stroke = criterion['first_stroke'](outputs['first_stroke'], first_strokes)
                    
                    # 增加笔画数预测的权重
                    loss = 2.0 * loss_stroke_count + loss_first_stroke
                    
                    # Update statistics
                    val_loss += loss.item() * images.size(0)
                    
                    # Calculate accuracy
                    _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
                    val_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
                    
                    # 分析每个类别的准确率
                    for pred, target in zip(stroke_count_preds.cpu().numpy(), stroke_counts.cpu().numpy()):
                        if target not in stroke_count_total_by_class:
                            stroke_count_total_by_class[target] = 0
                            stroke_count_correct_by_class[target] = 0
                        
                        stroke_count_total_by_class[target] += 1
                        if pred == target:
                            stroke_count_correct_by_class[target] += 1
                    
                    _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
                    val_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
                    
                    val_samples += images.size(0)
            
            # Calculate epoch statistics
            val_loss = val_loss / val_samples
            val_stroke_count_acc = val_stroke_count_correct / val_samples
            val_first_stroke_acc = val_first_stroke_correct / val_samples
            
            # 计算每个类别的准确率
            class_accuracies = {}
            for stroke_count in sorted(stroke_count_total_by_class.keys()):
                if stroke_count_total_by_class[stroke_count] > 0:
                    acc = stroke_count_correct_by_class[stroke_count] / stroke_count_total_by_class[stroke_count]
                    class_accuracies[f"stroke_count_{stroke_count}_acc"] = acc
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Stroke Count Acc: {train_stroke_count_acc:.4f}, First Stroke Acc: {train_first_stroke_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Stroke Count Acc: {val_stroke_count_acc:.4f}, First Stroke Acc: {val_first_stroke_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 打印每个类别的准确率
            print("  笔画数类别准确率:")
            for stroke_count in sorted(stroke_count_total_by_class.keys()):
                if stroke_count_total_by_class[stroke_count] > 0:
                    acc = stroke_count_correct_by_class[stroke_count] / stroke_count_total_by_class[stroke_count]
                    print(f"    笔画数 {stroke_count}: {acc:.4f} ({stroke_count_correct_by_class[stroke_count]}/{stroke_count_total_by_class[stroke_count]})")
            
            # Log metrics to MLflow
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_stroke_count_acc": train_stroke_count_acc,
                "val_stroke_count_acc": val_stroke_count_acc,
                "train_first_stroke_acc": train_first_stroke_acc,
                "val_first_stroke_acc": val_first_stroke_acc,
                "learning_rate": current_lr
            }
            
            # 添加每个类别的准确率
            metrics.update(class_accuracies)
            
            mlflow.log_metrics(metrics, step=epoch)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(model, "best_model_by_loss")
                
                # Save model locally
                torch.save(model.state_dict(), 'improved_model_best_loss.pth')
                
                # Reset early stopping counter
                counter = 0
            else:
                counter += 1
            
            # 同时保存笔画数准确率最高的模型
            if val_stroke_count_acc > best_stroke_count_acc:
                best_stroke_count_acc = val_stroke_count_acc
                print(f"  New best stroke count accuracy: {val_stroke_count_acc:.4f}")
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(model, "best_model_by_acc")
                
                # Save model locally
                torch.save(model.state_dict(), 'improved_model_best_acc.pth')
                
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'improved_model_final_{timestamp}.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to '{final_model_path}'")
        
        # Also save as the standard name for future runs
        torch.save(model.state_dict(), 'improved_model.pth')
        print(f"Final model also saved to 'improved_model.pth' for future runs")
        
        # Log final model as artifact
        mlflow.log_artifact(final_model_path)
        
        return model

if __name__ == "__main__":
    # Train model with improved strategies
    train_model_with_augmentation(
        max_chars=7000,
        augmentation_factor=10,
        batch_size=32,
        num_epochs=100,
        learning_rate=0.0003,
        use_balanced_dataset=True,
        use_focal_loss=True
    ) 