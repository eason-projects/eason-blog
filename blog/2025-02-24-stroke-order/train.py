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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

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
        self.current_sample = sample
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
        
        # Set the features dimension for stable-baselines3
        self.features_dim = 512
        
    def forward(self, observations):
        # Process image
        image_features = self.cnn(observations['image'].float())
        
        # Process stroke history
        stroke_features = self.stroke_encoder(observations['stroke_history'].float())
        
        # Combine features
        combined_features = torch.cat([image_features, stroke_features], dim=1)
        return self.combined(combined_features)

class StrokeOrderEvalCallback(BaseCallback):
    """Custom callback for evaluating stroke order prediction"""
    
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Use EvalCallback's evaluate method instead of custom implementation
            # Create a temporary environment for evaluation
            eval_env = StrokeOrderEnv()
            
            successes = 0
            total_rewards = 0
            stroke_accuracies = []
            
            for episode in range(self.n_eval_episodes):
                # Reset environment
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0
                correct_strokes = 0
                total_strokes = 0
                
                while not done:
                    # Get action from model
                    # Convert dict observation to a format the model can use
                    obs_dict = {
                        'image': np.expand_dims(obs['image'], axis=0),
                        'stroke_history': np.expand_dims(obs['stroke_history'], axis=0)
                    }
                    
                    action, _ = self.model.predict(obs_dict, deterministic=True)
                    
                    # Step environment with scalar action
                    obs, reward, terminated, truncated, info = eval_env.step(action[0])
                    
                    # Check if episode is done
                    done = terminated or truncated
                    
                    # Update metrics
                    episode_reward += reward
                    if reward > 0:
                        correct_strokes += 1
                    total_strokes += 1
                
                # Calculate episode metrics
                success = episode_reward > 0
                stroke_accuracy = correct_strokes / total_strokes if total_strokes > 0 else 0
                
                # Update overall metrics
                successes += int(success)
                total_rewards += episode_reward
                stroke_accuracies.append(stroke_accuracy)
            
            # Calculate mean metrics
            mean_reward = total_rewards / self.n_eval_episodes
            mean_success_rate = successes / self.n_eval_episodes
            mean_stroke_accuracy = sum(stroke_accuracies) / len(stroke_accuracies) if stroke_accuracies else 0
            
            # Log metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/success_rate", mean_success_rate)
            self.logger.record("eval/stroke_accuracy", mean_stroke_accuracy)
            
            # Print evaluation results
            if self.verbose > 0:
                print(f"\nStep {self.n_calls}")
                print(f"Eval success rate: {mean_success_rate:.2%}")
                print(f"Eval mean reward: {mean_reward:.2f}")
                print(f"Eval stroke accuracy: {mean_stroke_accuracy:.2%}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}")
                self.model.save("best_stroke_order_model")
            
            self.last_mean_reward = mean_reward
            
        return True

def train():
    # Create environments (one for training, one for evaluation)
    env = DummyVecEnv([lambda: StrokeOrderEnv()])
    eval_env = DummyVecEnv([lambda: StrokeOrderEnv()])
    
    # Create model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=0.0003)
    
    # Create callback
    eval_callback = StrokeOrderEvalCallback(
        eval_env=eval_env,
        eval_freq=1000,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Train model with callback
    model.learn(
        total_timesteps=100000,
        progress_bar=True,
        callback=eval_callback
    )
    
    # Save final model
    model.save("final_stroke_order_model")

def predict_strokes(model_path, num_samples=5):
    """Load a trained model and predict stroke order for random characters"""
    # Create environment
    env = StrokeOrderEnv()
    
    # Load model without specifying policy_kwargs
    model = PPO.load(model_path)
    
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    # Track overall statistics
    total_accuracy = 0
    stroke_distribution = {}
    
    for i in range(num_samples):
        # Reset environment to get a random character
        obs, _ = env.reset()
        done = False
        
        # Get character info - this is the actual Chinese character
        sample = env.current_sample
        char_symbol = sample['character']
        
        print(f"\nSample {i+1}: Character '{char_symbol}'")
        print(f"Target strokes: {env.target_strokes}")
        
        # Track predicted strokes
        predicted_strokes = []
        
        while not done:
            # Prepare observation for model - convert to PyTorch tensors
            obs_dict = {
                'image': torch.tensor(np.expand_dims(obs['image'], axis=0), dtype=torch.float32),
                'stroke_history': torch.tensor(np.expand_dims(obs['stroke_history'], axis=0), dtype=torch.float32)
            }
            
            # Get model prediction
            action, _states = model.predict(obs_dict, deterministic=True)
            
            # Convert action to stroke name
            stroke_name = env.stroke_types[action[0]]
            predicted_strokes.append(stroke_name)
            
            # Update stroke distribution statistics
            if stroke_name in stroke_distribution:
                stroke_distribution[stroke_name] += 1
            else:
                stroke_distribution[stroke_name] = 1
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            
            # Stop if we predict END token
            # if stroke_name == 'END':
            #     break
            if done:
                break
        
        print(f"Predicted strokes: {predicted_strokes}")
        
        # Calculate accuracy
        correct = 0
        for j, (pred, target) in enumerate(zip(predicted_strokes, env.target_strokes)):
            if pred == target:
                correct += 1
            else:
                break
                
        accuracy = correct / len(env.target_strokes) if len(env.target_strokes) > 0 else 0
        total_accuracy += accuracy
        print(f"Accuracy: {accuracy:.2%}")
        print("-" * 30)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average accuracy: {(total_accuracy / num_samples):.2%}")
    print("\nStroke distribution in predictions:")
    sorted_strokes = sorted(stroke_distribution.items(), key=lambda x: x[1], reverse=True)
    for stroke, count in sorted_strokes:
        print(f"  {stroke}: {count} times ({count/sum(stroke_distribution.values()):.2%})")

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

    # Uncomment one of these:
    # train()
    # predict_strokes("best_stroke_order_model", num_samples=10)
    predict_strokes("final_stroke_order_model", num_samples=10)
