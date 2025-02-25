import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from PIL import Image

class StrokeOrderEnv(gym.Env):
    """Custom Environment for Chinese character stroke order prediction"""
    
    def __init__(self, image_size=64):
        super().__init__()
        
        # 定义基本笔画类型（简化版本）
        self.stroke_types = ['横', '竖', '撇', '捺', '点', '折', 'END']
        
        # 动作空间：离散的笔画类型选择
        self.action_space = spaces.Discrete(len(self.stroke_types))
        
        # 状态空间：图像(channel=1, H=64, W=64) + 已选笔画序列的one-hot编码(max_strokes=20)
        self.image_size = image_size
        self.max_strokes = 20
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(1, image_size, image_size), dtype=np.uint8),
            'stroke_history': spaces.Box(low=0, high=1, shape=(self.max_strokes, len(self.stroke_types)), dtype=np.float32)
        })
        
        # 当前状态
        self.current_image = None
        self.current_strokes = []
        self.target_strokes = []
        
    def reset(self):
        """重置环境到初始状态"""
        # 这里应该从数据集加载一个新的汉字图片和其正确的笔画序列
        # 简化示例：假设我们有一个固定的测试用例
        self.current_image = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.current_strokes = []
        self.target_strokes = ['横', '竖', '撇', 'END']  # 示例目标序列
        
        return self._get_observation()
        
    def step(self, action):
        """执行一个动作并返回新的状态、奖励等"""
        done = False
        reward = 0
        
        # 获取选择的笔画
        selected_stroke = self.stroke_types[action]
        
        # 如果选择了END
        if selected_stroke == 'END':
            done = True
            # 如果所有笔画都正确且完成，给予额外奖励
            if len(self.current_strokes) == len(self.target_strokes) - 1:
                reward = 10
            else:
                reward = -5
        else:
            # 检查当前步骤是否正确
            current_step = len(self.current_strokes)
            if current_step < len(self.target_strokes) - 1:  # -1是因为不计算END
                if selected_stroke == self.target_strokes[current_step]:
                    reward = 1
                    self.current_strokes.append(selected_stroke)
                else:
                    reward = -1
                    done = True
            else:
                reward = -1
                done = True
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """构建当前状态的观察"""
        # 构建笔画历史的one-hot编码
        stroke_history = np.zeros((self.max_strokes, len(self.stroke_types)))
        for i, stroke in enumerate(self.current_strokes):
            if i >= self.max_strokes:
                break
            stroke_idx = self.stroke_types.index(stroke)
            stroke_history[i][stroke_idx] = 1
            
        return {
            'image': self.current_image,
            'stroke_history': stroke_history
        }

# 自定义特征提取器
class CustomCombinedExtractor(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        
        # CNN用于处理图像
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 线性层用于处理笔画历史
        self.stroke_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_space['stroke_history'].shape[0] * 
                     observation_space['stroke_history'].shape[1], 128),
            nn.ReLU()
        )
        
        # 组合特征
        self.combined = nn.Sequential(
            nn.Linear(16384 + 128, 512),  # 16384是CNN输出大小
            nn.ReLU()
        )
        
    def forward(self, observations):
        # 处理图像
        image_features = self.cnn(observations['image'].float())
        
        # 处理笔画历史
        stroke_features = self.stroke_encoder(observations['stroke_history'].float())
        
        # 组合特征
        combined_features = torch.cat([image_features, stroke_features], dim=1)
        return self.combined(combined_features)

def train():
    # 创建环境
    env = DummyVecEnv([lambda: StrokeOrderEnv()])
    
    # 创建模型
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=0.0003)
    
    # 训练模型
    model.learn(total_timesteps=100000)
    
    # 保存模型
    model.save("stroke_order_model")

if __name__ == "__main__":
    train()
