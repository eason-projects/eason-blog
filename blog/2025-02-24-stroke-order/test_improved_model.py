import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import os
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 导入数据集和模型
from pretrain_cnn import StrokeOrderPretrainDataset
from pretrain_cnn_resnet import ImprovedCNNModel

def test_model(model_path, max_chars=7000, batch_size=32):
    """测试改进后的模型性能"""
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 检查是否有MPS设备可用
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS设备进行测试")
    else:
        device = torch.device("cpu")
        print("MPS不可用，使用CPU")
    
    # 创建测试数据集
    test_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        max_chars=max_chars,
        train=False,
        split_ratio=0.9
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 分析测试集中的笔画数分布
    stroke_counts = [sample['stroke_count'] for sample in test_dataset.samples]
    stroke_count_counter = Counter(stroke_counts)
    print("测试集笔画数分布:", dict(sorted(stroke_count_counter.items())))
    
    # 获取笔画类型数量和最大笔画数
    max_stroke_count = max(stroke_counts)
    num_stroke_types = len(test_dataset.dataset.stroke_names)
    
    print(f"最大笔画数: {max_stroke_count}")
    print(f"笔画类型数量: {num_stroke_types}")
    
    # 创建模型
    model = ImprovedCNNModel(max_stroke_count, num_stroke_types)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 用于存储预测结果
    all_stroke_count_preds = []
    all_stroke_count_targets = []
    all_first_stroke_preds = []
    all_first_stroke_targets = []
    
    # 用于存储每个类别的准确率
    stroke_count_correct_by_class = {}
    stroke_count_total_by_class = {}
    
    # 用于存储错误预测的样本
    error_samples = []
    
    # 测试模型
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="测试模型"):
            # 获取样本
            sample = test_dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)  # 添加批次维度
            stroke_count = sample['stroke_count']
            first_stroke = sample['first_stroke']
            character = sample['character']
            
            # 前向传播
            outputs = model(image)
            
            # 获取预测结果
            _, stroke_count_pred = torch.max(outputs['stroke_count'], 1)
            _, first_stroke_pred = torch.max(outputs['first_stroke'], 1)
            
            # 转换为CPU上的numpy数组
            stroke_count_pred = stroke_count_pred.cpu().numpy()[0]
            first_stroke_pred = first_stroke_pred.cpu().numpy()[0]
            
            # 存储预测结果
            all_stroke_count_preds.append(stroke_count_pred)
            all_stroke_count_targets.append(stroke_count)
            all_first_stroke_preds.append(first_stroke_pred)
            all_first_stroke_targets.append(first_stroke)
            
            # 更新每个类别的统计信息
            if stroke_count not in stroke_count_total_by_class:
                stroke_count_total_by_class[stroke_count] = 0
                stroke_count_correct_by_class[stroke_count] = 0
            
            stroke_count_total_by_class[stroke_count] += 1
            if stroke_count_pred == stroke_count:
                stroke_count_correct_by_class[stroke_count] += 1
            else:
                # 存储错误预测的样本
                error_samples.append({
                    'image': sample['image'].numpy(),
                    'character': character,
                    'true_stroke_count': stroke_count,
                    'pred_stroke_count': stroke_count_pred
                })
    
    # 计算总体准确率
    stroke_count_acc = np.mean(np.array(all_stroke_count_preds) == np.array(all_stroke_count_targets))
    first_stroke_acc = np.mean(np.array(all_first_stroke_preds) == np.array(all_first_stroke_targets))
    
    print(f"笔画数预测准确率: {stroke_count_acc:.4f}")
    print(f"第一个笔画预测准确率: {first_stroke_acc:.4f}")
    
    # 打印每个类别的准确率
    print("\n每个笔画数类别的准确率:")
    for stroke_count in sorted(stroke_count_total_by_class.keys()):
        acc = stroke_count_correct_by_class[stroke_count] / stroke_count_total_by_class[stroke_count]
        print(f"  笔画数 {stroke_count}: {acc:.4f} ({stroke_count_correct_by_class[stroke_count]}/{stroke_count_total_by_class[stroke_count]})")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_stroke_count_targets, all_stroke_count_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测笔画数')
    plt.ylabel('真实笔画数')
    plt.title('笔画数预测混淆矩阵')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存到 'confusion_matrix.png'")
    
    # 生成分类报告
    report = classification_report(all_stroke_count_targets, all_stroke_count_preds, output_dict=True)
    print("\n分类报告:")
    for class_id, metrics in report.items():
        if class_id.isdigit():
            print(f"  笔画数 {class_id}:")
            print(f"    精确率: {metrics['precision']:.4f}")
            print(f"    召回率: {metrics['recall']:.4f}")
            print(f"    F1分数: {metrics['f1-score']:.4f}")
            print(f"    支持度: {metrics['support']}")
    
    # 可视化一些错误预测的样本
    num_errors_to_show = min(25, len(error_samples))
    if num_errors_to_show > 0:
        # 随机选择一些错误样本
        random.shuffle(error_samples)
        selected_errors = error_samples[:num_errors_to_show]
        
        # 创建图形
        rows = int(np.ceil(num_errors_to_show / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
        axes = axes.flatten()
        
        for i, error in enumerate(selected_errors):
            if i < num_errors_to_show:
                # 显示图像
                ax = axes[i]
                img = error['image'].squeeze()
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{error['character']}\n真实: {error['true_stroke_count']}, 预测: {error['pred_stroke_count']}")
                ax.axis('off')
        
        # 隐藏未使用的子图
        for i in range(num_errors_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('error_samples.png')
        print("错误预测样本已保存到 'error_samples.png'")
    
    # 绘制每个笔画数类别的准确率
    plt.figure(figsize=(12, 6))
    stroke_counts = sorted(stroke_count_total_by_class.keys())
    accuracies = [stroke_count_correct_by_class[sc] / stroke_count_total_by_class[sc] for sc in stroke_counts]
    sample_counts = [stroke_count_total_by_class[sc] for sc in stroke_counts]
    
    # 创建双轴图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制准确率柱状图
    bars = ax1.bar(stroke_counts, accuracies, alpha=0.7, color='blue')
    ax1.set_xlabel('笔画数')
    ax1.set_ylabel('准确率', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.0)
    
    # 在柱状图上添加准确率标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', color='blue')
    
    # 创建第二个Y轴，显示样本数量
    ax2 = ax1.twinx()
    ax2.plot(stroke_counts, sample_counts, 'r-', marker='o')
    ax2.set_ylabel('样本数量', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('每个笔画数类别的准确率和样本数量')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_by_stroke_count.png')
    print("每个笔画数类别的准确率图已保存到 'accuracy_by_stroke_count.png'")
    
    return stroke_count_acc, first_stroke_acc

if __name__ == "__main__":
    # 测试改进后的模型
    # 首先尝试加载按准确率保存的最佳模型
    if os.path.exists('improved_model_best_acc.pth'):
        print("使用按准确率保存的最佳模型进行测试")
        model_path = 'improved_model_best_acc.pth'
    # 如果不存在，尝试加载按损失保存的最佳模型
    elif os.path.exists('improved_model_best_loss.pth'):
        print("使用按损失保存的最佳模型进行测试")
        model_path = 'improved_model_best_loss.pth'
    # 如果都不存在，使用最终保存的模型
    elif os.path.exists('improved_model.pth'):
        print("使用最终保存的模型进行测试")
        model_path = 'improved_model.pth'
    else:
        print("找不到改进后的模型，请先训练模型")
        exit(1)
    
    test_model(model_path, max_chars=7000) 