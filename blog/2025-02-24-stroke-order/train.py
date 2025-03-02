import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Import the pretrained CNN model
from dataset import StrokeOrderDataset, PretrainedCombinedExtractor
from models import MobileNetV3Model

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
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create environments (one for training, one for evaluation)
    env = DummyVecEnv([lambda: StrokeOrderEnv()])
    eval_env = DummyVecEnv([lambda: StrokeOrderEnv()])
    
    # Create model with pretrained feature extractor
    policy_kwargs = dict(
        features_extractor_class=PretrainedCombinedExtractor,
        # You can pass additional arguments to the feature extractor
        features_extractor_kwargs=dict(
            pretrained_model_path='augmented_model.pth'  # Path to the pretrained model
        )
    )
    
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, 
                verbose=1, learning_rate=0.0003, device=device)
    
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
    train()
    # predict_strokes("best_stroke_order_model", num_samples=10)
    # predict_strokes("final_stroke_order_model", num_samples=10)
