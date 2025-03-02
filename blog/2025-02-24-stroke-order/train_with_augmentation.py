import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import mlflow
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 导入数据集和模型
from pretrain_cnn import StrokeOrderPretrainDataset
from pretrain_cnn_resnet import ImprovedCNNModel, AugmentedStrokeOrderDataset, visualize_augmentations, BalancedStrokeOrderDataset, FocalLoss, WeightedCrossEntropyLoss

def train_with_augmentation(args):
    """使用数据增强训练改进的模型"""
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设置MLflow跟踪URI
    if args.use_mlflow:
        mlflow.set_tracking_uri("http://localhost:8090")
        experiment_name = "stroke_order_cnn_augmented"
        mlflow.set_experiment(experiment_name)
    
    # 检查是否有MPS设备可用
    if torch.backends.mps.is_available() and not args.force_cpu:
        device = torch.device("mps")
        print("使用MPS设备进行训练")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")
    
    # 创建基础数据集
    train_base_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        max_chars=args.max_chars,
        train=True,
        split_ratio=0.9
    )
    
    val_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        max_chars=args.max_chars,
        train=False,
        split_ratio=0.9
    )
    
    # 创建训练数据集
    if args.use_balanced_dataset:
        print("使用平衡采样数据集")
        train_dataset = BalancedStrokeOrderDataset(train_base_dataset)
    else:
        print("使用数据增强数据集")
        train_dataset = AugmentedStrokeOrderDataset(
            base_dataset=train_base_dataset,
            augmentation_factor=args.augmentation_factor
        )
    
    print(f"基础训练集大小: {len(train_base_dataset)}")
    print(f"最终训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 可视化增强效果
    if not args.use_balanced_dataset and args.visualize_augmentations:
        visualize_augmentations(train_dataset, num_samples=5)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 获取笔画类型数量和最大笔画数
    max_stroke_count = max(sample['stroke_count'] for sample in train_base_dataset.samples)
    num_stroke_types = len(train_base_dataset.dataset.stroke_names)
    
    print(f"最大笔画数: {max_stroke_count}")
    print(f"笔画类型数量: {num_stroke_types}")
    
    # 创建改进的模型
    model = ImprovedCNNModel(max_stroke_count, num_stroke_types)
    
    # 计算笔画数类别权重
    if args.use_weighted_loss or args.use_focal_loss:
        stroke_counts = [sample['stroke_count'] for sample in train_base_dataset.samples]
        from collections import Counter
        stroke_count_counter = Counter(stroke_counts)
        print("训练集笔画数分布:", dict(sorted(stroke_count_counter.items())))
        
        # 计算类别权重（反比于样本数量）
        total_samples = len(stroke_counts)
        class_weights = {}
        for stroke_count, count in stroke_count_counter.items():
            class_weights[stroke_count] = total_samples / (count * len(stroke_count_counter))
        
        # 转换为张量
        weight_tensor = torch.ones(max_stroke_count + 1)
        for stroke_count, weight in class_weights.items():
            weight_tensor[stroke_count] = weight
        
        print("笔画数类别权重:", weight_tensor)
        weight_tensor = weight_tensor.to(device)
    else:
        weight_tensor = None
    
    # 定义损失函数和优化器
    if args.use_focal_loss:
        print("使用焦点损失函数")
        criterion = {
            'stroke_count': FocalLoss(gamma=args.focal_gamma, weight=weight_tensor),
            'first_stroke': nn.CrossEntropyLoss()
        }
    elif args.use_weighted_loss:
        print("使用加权交叉熵损失函数")
        criterion = {
            'stroke_count': WeightedCrossEntropyLoss(weight=weight_tensor),
            'first_stroke': nn.CrossEntropyLoss()
        }
    else:
        print("使用标准交叉熵损失函数")
        criterion = {
            'stroke_count': nn.CrossEntropyLoss(),
            'first_stroke': nn.CrossEntropyLoss()
        }
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate/100
    )
    
    model.to(device)
    
    # 记录参数到MLflow
    if args.use_mlflow:
        with mlflow.start_run():
            mlflow.log_params({
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_chars": args.max_chars,
                "weight_decay": args.weight_decay,
                "max_stroke_count": max_stroke_count,
                "num_stroke_types": num_stroke_types,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "augmentation_factor": args.augmentation_factor if not args.use_balanced_dataset else "N/A",
                "use_balanced_dataset": args.use_balanced_dataset,
                "use_focal_loss": args.use_focal_loss,
                "focal_gamma": args.focal_gamma if args.use_focal_loss else "N/A",
                "use_weighted_loss": args.use_weighted_loss,
                "stroke_count_loss_weight": args.stroke_count_loss_weight,
                "device": str(device),
                "model_type": "ImprovedCNNModel"
            })
            
            # 训练模型
            best_model = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, mlflow
            )
    else:
        # 训练模型（不使用MLflow）
        best_model = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, None
        )
    
    return best_model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args, mlflow_client=None):
    """训练模型的核心函数"""
    
    # 最佳模型状态
    best_val_loss = float('inf')
    best_model_state = None
    best_stroke_count_acc = 0.0
    
    # 早停参数
    patience = args.patience
    counter = 0
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_stroke_count_acc': [],
        'val_stroke_count_acc': [],
        'train_first_stroke_acc': [],
        'val_first_stroke_acc': [],
        'learning_rate': []
    }
    
    # 训练循环
    for epoch in range(args.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_stroke_count_correct = 0
        train_first_stroke_correct = 0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]"):
            # 将数据移动到设备
            images = batch['image'].to(device)
            stroke_counts = batch['stroke_count'].to(device)
            first_strokes = batch['first_stroke'].to(device)
            
            # 清零参数梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_stroke_count = criterion['stroke_count'](outputs['stroke_count'], stroke_counts)
            loss_first_stroke = criterion['first_stroke'](outputs['first_stroke'], first_strokes)
            
            # 增加笔画数预测的权重
            loss = args.stroke_count_loss_weight * loss_stroke_count + loss_first_stroke
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item() * images.size(0)
            
            # 计算准确率
            _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
            train_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
            
            _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
            train_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
            
            train_samples += images.size(0)
        
        # 计算训练统计信息
        train_loss = train_loss / train_samples
        train_stroke_count_acc = train_stroke_count_correct / train_samples
        train_first_stroke_acc = train_first_stroke_correct / train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_stroke_count_correct = 0
        val_first_stroke_correct = 0
        val_samples = 0
        
        # 用于分析每个笔画数类别的准确率
        stroke_count_correct_by_class = {}
        stroke_count_total_by_class = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]"):
                # 将数据移动到设备
                images = batch['image'].to(device)
                stroke_counts = batch['stroke_count'].to(device)
                first_strokes = batch['first_stroke'].to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss_stroke_count = criterion['stroke_count'](outputs['stroke_count'], stroke_counts)
                loss_first_stroke = criterion['first_stroke'](outputs['first_stroke'], first_strokes)
                
                # 增加笔画数预测的权重
                loss = args.stroke_count_loss_weight * loss_stroke_count + loss_first_stroke
                
                # 更新统计信息
                val_loss += loss.item() * images.size(0)
                
                # 计算准确率
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
        
        # 计算验证统计信息
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
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_stroke_count_acc'].append(train_stroke_count_acc)
        history['val_stroke_count_acc'].append(val_stroke_count_acc)
        history['train_first_stroke_acc'].append(train_first_stroke_acc)
        history['val_first_stroke_acc'].append(val_first_stroke_acc)
        history['learning_rate'].append(current_lr)
        
        # 打印统计信息
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Stroke Count Acc: {train_stroke_count_acc:.4f}, First Stroke Acc: {train_first_stroke_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Stroke Count Acc: {val_stroke_count_acc:.4f}, First Stroke Acc: {val_first_stroke_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # 打印每个类别的准确率
        print("  笔画数类别准确率:")
        for stroke_count in sorted(stroke_count_total_by_class.keys()):
            if stroke_count_total_by_class[stroke_count] > 0:
                acc = stroke_count_correct_by_class[stroke_count] / stroke_count_total_by_class[stroke_count]
                print(f"    笔画数 {stroke_count}: {acc:.4f} ({stroke_count_correct_by_class[stroke_count]}/{stroke_count_total_by_class[stroke_count]})")
        
        # 记录指标到MLflow
        if mlflow_client is not None:
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
            
            mlflow_client.log_metrics(metrics, step=epoch)
        
        # 保存基于验证损失的最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            
            # 记录最佳模型到MLflow
            if mlflow_client is not None:
                mlflow_client.pytorch.log_model(model, "best_model_by_loss")
            
            # 本地保存模型
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_by_loss.pth'))
            
            # 重置早停计数器
            counter = 0
        else:
            counter += 1
        
        # 同时保存笔画数准确率最高的模型
        if val_stroke_count_acc > best_stroke_count_acc:
            best_stroke_count_acc = val_stroke_count_acc
            print(f"  New best stroke count accuracy: {val_stroke_count_acc:.4f}")
            
            # 记录最佳模型到MLflow
            if mlflow_client is not None:
                mlflow_client.pytorch.log_model(model, "best_model_by_acc")
            
            # 本地保存模型
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_by_acc.pth'))
        
        # 早停
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 保存带时间戳的最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(args.output_dir, f'final_model_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to '{final_model_path}'")
    
    # 同时保存为标准名称，用于未来运行
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Final model also saved to '{os.path.join(args.output_dir, 'final_model.pth')}'")
    
    # 记录最终模型作为MLflow工件
    if mlflow_client is not None:
        mlflow_client.log_artifact(final_model_path)
    
    # 绘制训练历史
    plot_training_history(history, args.output_dir)
    
    return model

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    
    # 绘制笔画数准确率
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_stroke_count_acc'], label='Train Accuracy')
    plt.plot(history['val_stroke_count_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Stroke Count Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'stroke_count_acc_history.png'))
    
    # 绘制第一个笔画准确率
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_first_stroke_acc'], label='Train Accuracy')
    plt.plot(history['val_first_stroke_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('First Stroke Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'first_stroke_acc_history.png'))
    
    # 绘制学习率
    plt.figure(figsize=(10, 5))
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_rate_history.png'))
    
    print(f"Training history plots saved to '{output_dir}'")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train stroke order prediction model with augmentation')
    
    # 数据参数
    parser.add_argument('--max_chars', type=int, default=7000, help='Maximum number of characters to use')
    parser.add_argument('--augmentation_factor', type=int, default=12, help='Number of augmented versions to create for each sample')
    parser.add_argument('--use_balanced_dataset', action='store_true', help='Use balanced dataset instead of augmentation')
    parser.add_argument('--visualize_augmentations', action='store_true', help='Visualize augmentations')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 损失函数参数
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted loss')
    parser.add_argument('--stroke_count_loss_weight', type=float, default=2.0, help='Weight for stroke count loss')
    
    # 优化器参数
    parser.add_argument('--clip_grad_norm', action='store_true', help='Clip gradient norm')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--use_mlflow', action='store_true', help='Use MLflow for tracking')
    parser.add_argument('--force_cpu', action='store_true', help='Force using CPU even if MPS is available')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    train_with_augmentation(args) 