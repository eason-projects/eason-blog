# 汉字笔顺预测模型

这个项目实现了一个基于深度学习的汉字笔顺预测模型，可以预测汉字的笔画数量和第一个笔画类型。

## 项目结构

- `pretrain_cnn.py`: 原始CNN模型和数据集定义
- `pretrain_cnn_resnet.py`: 改进的ResNet模型、数据增强和平衡采样实现
- `train_with_augmentation.py`: 使用数据增强训练改进模型的脚本
- `test_improved_model.py`: 测试改进模型性能的脚本
- `train.py`: 使用强化学习训练笔顺预测模型的脚本
- `run_mlflow_server.py`: 启动MLflow服务器的脚本
- `load_from_mlflow.py`: 从MLflow加载和评估模型的脚本

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm
- scikit-learn
- MLflow
- Gymnasium
- Stable-Baselines3

## 数据准备

确保以下文件在工作目录中：
- `stroke-order-jian.json`: 汉字笔顺数据
- `stroke-table.json`: 笔画类型表
- `images/`: 包含汉字图像的文件夹

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动MLflow服务器

在训练或评估模型之前，启动MLflow服务器以跟踪实验：

```bash
python run_mlflow_server.py
```

MLflow UI将在 http://localhost:8090 可访问。

### 训练模型

#### 使用数据增强训练CNN模型：

```bash
python train_with_augmentation.py --max_chars 7000 --augmentation_factor 12 --batch_size 32 --num_epochs 100 --learning_rate 0.0003 --use_focal_loss --use_mlflow --stroke_count_loss_weight 2.0 --output_dir ./output
```

#### 使用强化学习训练笔顺预测模型：

```bash
# 从头开始训练新模型
python train.py --mode train

# 从现有模型继续训练
python train.py --mode train --model best_stroke_order_model --timesteps 50000 --lr 0.0001

# 使用自定义输出模型名称
python train.py --mode train --output my_custom_model_name

# 预测模式
python train.py --mode predict --model best_stroke_order_model --samples 20

# 测试数据集
python train.py --mode test_dataset
```

训练过程中的指标、参数和模型将自动记录到MLflow中。

#### 主要参数说明：

- `--mode`: 运行模式，可选 'train'（训练）, 'predict'（预测）或 'test_dataset'（测试数据集）
- `--model`: 用于继续训练或预测的现有模型路径
- `--samples`: 预测模式下的样本数量
- `--timesteps`: 训练的总时间步数
- `--lr`: 学习率
- `--output`: 输出模型的自定义名称（不含.zip扩展名）
- `--max_chars`: 使用的最大字符数量
- `--augmentation_factor`: 每个样本创建的增强版本数量
- `--use_balanced_dataset`: 使用平衡采样数据集而非数据增强
- `--visualize_augmentations`: 可视化数据增强效果
- `--batch_size`: 批量大小
- `--num_epochs`: 训练轮数
- `--use_focal_loss`: 使用焦点损失函数
- `--use_weighted_loss`: 使用加权交叉熵损失函数
- `--stroke_count_loss_weight`: 笔画数损失的权重
- `--use_mlflow`: 使用MLflow跟踪训练过程

### 测试模型

#### 测试CNN模型：

```bash
python test_improved_model.py --model_path ./output/best_model_by_acc.pth --max_chars 1000 --batch_size 32
```

#### 测试强化学习模型：

```bash
# 修改train.py中的代码，取消注释predict_strokes行
python train.py
```

#### 从MLflow加载和评估模型：

```bash
# 列出可用的MLflow运行
python load_from_mlflow.py --list-runs

# 加载并评估特定运行的模型
python load_from_mlflow.py --run-id <RUN_ID>
```

## MLflow集成

本项目使用MLflow进行实验跟踪和模型管理：

1. **实验跟踪**：
   - 自动记录训练参数（学习率、批量大小等）
   - 记录评估指标（奖励、成功率、笔画准确率）
   - 可视化训练进度

2. **模型管理**：
   - 保存最佳模型和最终模型
   - 记录模型性能指标
   - 方便模型比较和选择

3. **预测结果记录**：
   - 记录每个样本的预测结果
   - 计算并记录整体准确率
   - 分析笔画分布

<!-- 有关MLflow集成的详细信息，请参阅 [MLFLOW_README.md](./MLFLOW_README.md)。 -->

## 改进策略

本项目实现了以下改进策略来提高笔画数预测准确率：

1. **模型架构改进**：
   - 使用预训练的ResNet18作为特征提取器
   - 添加更深的全连接层
   - 增加Dropout层减少过拟合

2. **损失函数优化**：
   - 实现加权交叉熵损失
   - 实现焦点损失，关注难以分类的样本
   - 增加笔画数预测在总损失中的权重

3. **数据平衡策略**：
   - 实现平衡采样，确保各笔画数类别样本均衡
   - 基于频率计算类别权重

4. **数据增强技术**：
   - 旋转、平移、缩放、剪切
   - 模糊、弹性变形
   - 添加噪声
   - 腐蚀、膨胀
   - 组合增强

5. **训练策略优化**：
   - 使用AdamW优化器
   - 余弦退火学习率调度
   - 梯度裁剪
   - 基于准确率保存模型

6. **强化学习方法**：
   - 使用PPO算法训练笔顺预测模型
   - 自定义环境和奖励函数
   - 使用预训练CNN模型作为特征提取器

## 结果分析

训练过程会生成以下可视化结果：

- `loss_history.png`: 训练和验证损失曲线
- `stroke_count_acc_history.png`: 笔画数准确率曲线
- `first_stroke_acc_history.png`: 第一笔画准确率曲线
- `learning_rate_history.png`: 学习率变化曲线

测试脚本会生成：

- `confusion_matrix.png`: 笔画数预测的混淆矩阵
- `accuracy_by_stroke_count.png`: 各笔画数类别的准确率
- `error_samples.png`: 错误预测样例可视化

此外，所有训练和评估结果都可以在MLflow UI中查看和比较。

## 许可证

MIT 