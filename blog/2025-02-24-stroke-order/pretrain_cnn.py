import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Import the dataset class from train.py
from train import StrokeOrderDataset, CustomCombinedExtractor

class StrokeOrderPretrainDataset(Dataset):
    """Dataset for pretraining the CNN on stroke count prediction"""
    
    def __init__(self, stroke_order_path, stroke_table_path, image_folder, max_chars=1000, train=True):
        # Create the original dataset
        self.dataset = StrokeOrderDataset(
            stroke_order_path=stroke_order_path,
            stroke_table_path=stroke_table_path,
            image_folder=image_folder,
            max_chars=max_chars
        )
        
        # Split characters into train/test sets (80/20 split)
        all_chars = self.dataset.characters
        random.seed(42)  # For reproducibility
        random.shuffle(all_chars)
        
        split_idx = int(len(all_chars) * 0.8)
        if train:
            self.chars = all_chars[:split_idx]
        else:
            self.chars = all_chars[split_idx:]
            
        # Create samples
        self.samples = []
        for char in self.chars:
            stroke_seq_codes = self.dataset.char_strokes[char]
            
            # Convert stroke sequence to full stroke information
            strokes = []
            for code in stroke_seq_codes:
                stroke_info = self.dataset.get_stroke_info(code)
                if stroke_info:
                    strokes.append(stroke_info['name'])
            
            # Load image
            image_path = os.path.join(self.dataset.image_folder, f"{len(strokes)}_{char}.png")
            try:
                image = Image.open(image_path).convert('L')  # Convert to grayscale
                image = np.array(image)
                if image.shape != (64, 64):
                    image = Image.fromarray(image).resize((64, 64))
                    image = np.array(image)
                image = image.reshape(1, 64, 64)  # Add channel dimension
                
                # Add sample
                self.samples.append({
                    'image': image,
                    'stroke_count': len(strokes),
                    'first_stroke': self.dataset.stroke_names.index(strokes[0]) if strokes else 0,
                    'character': char
                })
            except FileNotFoundError:
                print(f"Warning: Image not found for character {char}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'image': torch.tensor(sample['image'], dtype=torch.float32),
            'stroke_count': sample['stroke_count'],
            'first_stroke': sample['first_stroke'],
            'character': sample['character']
        }

class CNNModel(nn.Module):
    """CNN model for stroke count prediction"""
    
    def __init__(self, num_strokes, num_classes):
        super().__init__()
        
         # CNN for processing images - same architecture as in CustomCombinedExtractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Additional conv layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # Further reduce spatial dimensions
            nn.Flatten()
        )
        
        # Update CNN output size (8x8x128 = 8192)
        self.cnn_output_size = 8192
        
        # Prediction heads
        self.stroke_count_head = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_strokes + 1)  # +1 for 0 strokes
        )
        
        self.first_stroke_head = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
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

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    """Train the CNN model"""
    
    # Define loss function and optimizer
    criterion = {
        'stroke_count': nn.CrossEntropyLoss(),
        'first_stroke': nn.CrossEntropyLoss()
    }
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_stroke_count_acc': [],
        'val_stroke_count_acc': [],
        'train_first_stroke_acc': [],
        'val_first_stroke_acc': []
    }
    
    # Best model state
    best_val_loss = float('inf')
    best_model_state = None
    
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
            loss = loss_stroke_count + loss_first_stroke
            
            # Backward pass and optimize
            loss.backward()
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
                loss = loss_stroke_count + loss_first_stroke
                
                # Update statistics
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
                val_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
                
                _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
                val_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
                
                val_samples += images.size(0)
        
        # Calculate epoch statistics
        val_loss = val_loss / val_samples
        val_stroke_count_acc = val_stroke_count_correct / val_samples
        val_first_stroke_acc = val_first_stroke_correct / val_samples
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Stroke Count Acc: {train_stroke_count_acc:.4f}, First Stroke Acc: {train_first_stroke_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Stroke Count Acc: {val_stroke_count_acc:.4f}, First Stroke Acc: {val_first_stroke_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_stroke_count_acc'].append(train_stroke_count_acc)
        history['val_stroke_count_acc'].append(val_stroke_count_acc)
        history['train_first_stroke_acc'].append(train_first_stroke_acc)
        history['val_first_stroke_acc'].append(val_first_stroke_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history

def plot_history(history):
    """Plot training history"""
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot loss
    axs[0, 0].plot(history['train_loss'], label='Train')
    axs[0, 0].plot(history['val_loss'], label='Val')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot stroke count accuracy
    axs[0, 1].plot(history['train_stroke_count_acc'], label='Train')
    axs[0, 1].plot(history['val_stroke_count_acc'], label='Val')
    axs[0, 1].set_title('Stroke Count Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    
    # Plot first stroke accuracy
    axs[1, 0].plot(history['train_first_stroke_acc'], label='Train')
    axs[1, 0].plot(history['val_first_stroke_acc'], label='Val')
    axs[1, 0].set_title('First Stroke Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('pretrain_history.png')
    plt.close()

def save_cnn_for_rl(model, output_path='pretrained_cnn.pth'):
    """Save CNN weights for use in RL model"""
    
    # Extract CNN weights
    cnn_state_dict = {}
    for name, param in model.state_dict().items():
        if name.startswith('cnn.'):
            cnn_state_dict[name] = param
    
    # Save weights
    torch.save(cnn_state_dict, output_path)
    print(f"CNN weights saved to {output_path}")

def load_pretrained_cnn_to_rl_model(cnn_weights_path, observation_space):
    """Create a CustomCombinedExtractor with pretrained CNN weights"""
    
    # Create model
    model = CustomCombinedExtractor(observation_space)
    
    # Load CNN weights
    cnn_state_dict = torch.load(cnn_weights_path)
    
    # Update model weights
    model_state_dict = model.state_dict()
    for name, param in cnn_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
    
    model.load_state_dict(model_state_dict)
    
    return model

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    # Create datasets
    train_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        train=True
    )
    
    val_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        train=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Get number of stroke types
    max_stroke_count = max(sample['stroke_count'] for sample in train_dataset.samples)
    num_stroke_types = len(train_dataset.dataset.stroke_names)
    
    print(f"Max stroke count: {max_stroke_count}")
    print(f"Number of stroke types: {num_stroke_types}")
    
    # Create model
    model = CNNModel(max_stroke_count, num_stroke_types)
    model.to(device)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=20,
        learning_rate=0.001
    )
    
    # Plot training history
    plot_history(history)
    
    # Save model
    torch.save(model.state_dict(), 'pretrained_model.pth')
    print("Full model saved to pretrained_model.pth")
    
    # Save CNN weights for RL model
    save_cnn_for_rl(model)
    
    # Test loading pretrained CNN into RL model
    from gymnasium import spaces
    import numpy as np
    
    # Create dummy observation space
    observation_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8),
        'stroke_history': spaces.Box(low=0, high=num_stroke_types-1, shape=(30,), dtype=np.int64)
    })
    
    # Load pretrained CNN into RL model
    rl_model = load_pretrained_cnn_to_rl_model('pretrained_cnn.pth', observation_space)
    print("Successfully loaded pretrained CNN into RL model")

if __name__ == "__main__":
    main()