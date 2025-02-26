import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from PIL import Image
import json
import random
import os

class StrokeOrderDataset:
    """Dataset for Chinese character stroke order training"""
    
    def __init__(self, stroke_order_path, stroke_table_path, image_folder, max_chars=1000):
        # Load stroke table data
        with open(stroke_table_path, 'r', encoding='utf-8') as f:
            self.stroke_table = json.load(f)
            
        # Load character stroke data
        with open(stroke_order_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Get first max_chars items
            self.char_strokes = {k: v for i, (k, v) in enumerate(data.items()) if i < max_chars}
        
        self.image_folder = image_folder
        self.characters = list(self.char_strokes.keys())
        
        # Create list of all possible stroke names
        self.stroke_names = sorted(list(set(
            stroke_info['name'] 
            for stroke_info in self.stroke_table.values()
        )))
        self.stroke_names.append('END')  # Add END token
        
    def get_stroke_info(self, stroke_code):
        """Get full stroke information from stroke code"""
        if stroke_code in self.stroke_table:
            return self.stroke_table[stroke_code]
        return None
    
    def get_random_sample(self):
        """Get a random character with its image and stroke sequence"""
        char = random.choice(self.characters)
        stroke_seq_codes = self.char_strokes[char]
        
        # Convert stroke sequence to full stroke information
        strokes = []
        for code in stroke_seq_codes:
            stroke_info = self.get_stroke_info(code)
            if stroke_info:
                strokes.append(stroke_info['name'])
        strokes.append('END')  # Add END token
        
        # Load image
        image_path = os.path.join(self.image_folder, f"{len(strokes)-1}_{char}.png")
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = np.array(image)
            if image.shape != (64, 64):
                image = Image.fromarray(image).resize((64, 64))
                image = np.array(image)
            image = image.reshape(1, 64, 64)  # Add channel dimension
        except FileNotFoundError:
            print(f"Warning: Image not found for character {char}")
            image = np.zeros((1, 64, 64), dtype=np.uint8)
            
        return {
            'image': image,
            'strokes': strokes,
            'character': char,
            'stroke_count': len(strokes) - 1  # Exclude END token
        }

class StrokeOrderEnv(gym.Env):
    """Custom Environment for Chinese character stroke order prediction"""
    
    def __init__(self, image_size=64):
        super().__init__()
        
        # Initialize dataset
        self.dataset = StrokeOrderDataset(
            stroke_order_path='./stroke-order-jian.json',
            stroke_table_path='./stroke-table.json',
            image_folder='./images'
        )
        
        # Define action space using all possible stroke names
        self.stroke_types = self.dataset.stroke_names
        
        self.max_steps = 30  # Maximum number of steps per episode
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self.stroke_types))
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(1, image_size, image_size), dtype=np.uint8),
            'stroke_history': gym.spaces.Box(low=0, high=len(self.stroke_types)-1, shape=(self.max_steps,), dtype=np.int64)
        })
        
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Get random character data
        sample = self.dataset.get_random_sample()
        self.current_image = sample['image']
        self.target_strokes = sample['strokes']
        # Initialize stroke history with -1 (no stroke)
        self.current_strokes = np.full(self.max_steps, -1, dtype=np.int64)
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation state"""
        return {
            'image': self.current_image,
            'stroke_history': self.current_strokes
        }
    
    def step(self, action):
        """Execute one environment step"""
        # Record the action in stroke history
        self.current_strokes[self.current_step] = action
        self.current_step += 1
        
        # Convert action index to stroke name
        stroke = self.stroke_types[action]
        
        # Check if stroke matches target
        correct_stroke = (self.current_step <= len(self.target_strokes) and 
                         stroke == self.target_strokes[self.current_step-1])
        
        # Calculate reward
        if correct_stroke:
            reward = 1.0
        else:
            reward = -1.0
        
        # Check if episode is done
        done = (self.current_step >= len(self.target_strokes) or 
                self.current_step >= self.max_steps or 
                not correct_stroke)
        
        return self._get_observation(), reward, done, False, {
            'target_strokes': self.target_strokes,
            'current_strokes': [self.stroke_types[i] for i in self.current_strokes if i >= 0]
        }

# Custom feature extractor
class CustomCombinedExtractor(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        
        # CNN for processing images
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Linear layer for processing stroke history
        stroke_history_size = observation_space['stroke_history'].shape[0]
        self.stroke_encoder = nn.Sequential(
            nn.Linear(stroke_history_size, 128),
            nn.ReLU()
        )
        
        # Combine features
        self.combined = nn.Sequential(
            nn.Linear(16384 + 128, 512),  # 16384 is CNN output size
            nn.ReLU()
        )
        
    def forward(self, observations):
        # Process image
        image_features = self.cnn(observations['image'].float())
        
        # Process stroke history
        stroke_features = self.stroke_encoder(observations['stroke_history'].float())
        
        # Combine features
        combined_features = torch.cat([image_features, stroke_features], dim=1)
        return self.combined(combined_features)

def train():
    # Create environment
    env = DummyVecEnv([lambda: StrokeOrderEnv()])
    
    # Create model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=0.0003)
    
    # Train model
    model.learn(total_timesteps=100000)
    
    # Save model
    model.save("stroke_order_model")

if __name__ == "__main__":
    def test_dataset():
        print("Testing dataset...")
        dataset = StrokeOrderDataset(
            stroke_order_path='./stroke-order-jian.json',
            stroke_table_path='./stroke-table.json',
            image_folder='./images'
        )
        print("Dataset loaded")
        
        sample = dataset.get_random_sample()
        print(sample.keys())
        print(sample['image'].shape)
        print(sample['strokes'])
        print(sample['character'])
        print(sample['stroke_count'])

    # test_dataset()

    train()
