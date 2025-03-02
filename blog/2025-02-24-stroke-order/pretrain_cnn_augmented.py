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

# Import the dataset class and model from pretrain_cnn.py
from pretrain_cnn import MobileNetV3LiteModel, MobileNetV3Model, StrokeOrderPretrainDataset, CNNModel

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
            "dilation"            # 9: Dilation - thickens strokes
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
            # Rotation ±10 degrees
            angle = random.uniform(-10, 10)
            augmented_pil = image_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        
        elif aug_type == "slight_translation":
            # Translation up to 10% in each direction
            width, height = image_pil.size
            dx = int(random.uniform(-0.1, 0.1) * width)
            dy = int(random.uniform(-0.1, 0.1) * height)
            augmented_pil = ImageOps.expand(image_pil, border=(0, 0, 0, 0), fill=255)
            augmented_pil = augmented_pil.transform(
                (width, height),
                Image.AFFINE,
                (1, 0, dx, 0, 1, dy),
                resample=Image.BILINEAR,
                fillcolor=255
            )
        
        elif aug_type == "slight_scaling":
            # Scaling 90-110%
            width, height = image_pil.size
            scale = random.uniform(0.9, 1.1)
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
            shear_factor = random.uniform(-0.1, 0.1)
            augmented_pil = image_pil.transform(
                (width, height),
                Image.AFFINE,
                (1, shear_factor, 0, 0, 1, 0),
                resample=Image.BILINEAR,
                fillcolor=255
            )
        
        elif aug_type == "slight_blur":
            # Slight Gaussian blur
            augmented_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        
        elif aug_type == "elastic_deform":
            # Elastic deformation - simulates natural handwriting variations
            # Convert to binary image
            threshold = 128
            binary_np = np.array(image_pil) < threshold
            
            # Parameters for elastic deformation
            alpha = random.uniform(5, 10)  # Intensity of deformation
            sigma = random.uniform(3, 5)   # Smoothness of deformation
            
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
            pepper_prob = random.uniform(0.001, 0.01)
            pepper_mask = np.random.random(augmented_np.shape) < pepper_prob
            augmented_np[pepper_mask] = 0
            
            # Add random white dots (salt) - less noticeable on white background
            salt_prob = random.uniform(0.001, 0.005)
            salt_mask = np.random.random(augmented_np.shape) < salt_prob
            augmented_np[salt_mask] = 255
            
            augmented_pil = Image.fromarray(augmented_np)
        
        elif aug_type == "erosion":
            # Erosion - thins the strokes
            augmented_pil = image_pil.filter(ImageFilter.MinFilter(3))
        
        elif aug_type == "dilation":
            # Dilation - thickens the strokes
            augmented_pil = image_pil.filter(ImageFilter.MaxFilter(3))
        
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
    fig, axes = plt.subplots(num_samples, dataset.augmentation_factor, figsize=(15, 3 * num_samples))
    
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

def train_model_with_augmentation(max_chars=7000, augmentation_factor=10, batch_size=32, num_epochs=50, learning_rate=0.001):
    """Train the CNN model with data augmentation"""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:8090")
    
    # Set experiment name
    experiment_name = "stroke_order_cnn_augmented"
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
        
        # Create augmented training dataset
        train_dataset = AugmentedStrokeOrderDataset(
            base_dataset=train_base_dataset,
            augmentation_factor=augmentation_factor
        )
        
        print(f"Base train dataset size: {len(train_base_dataset)}")
        print(f"Augmented train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Visualize augmentations
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
        
        # Create model
        # model = CNNModel(max_stroke_count, num_stroke_types)
        model = MobileNetV3Model(max_stroke_count, num_stroke_types)
        # model = MobileNetV3LiteModel(max_stroke_count, num_stroke_types)

        augmented_model_path = 'augmented_model_mobilenetv3.pth'
        # augmented_model_path = 'augmented_model_mobilenetv3_lite.pth'

        if os.path.exists(augmented_model_path):
            print(f"Loading previously augmented model from {augmented_model_path} as starting point")
            model.load_state_dict(torch.load(augmented_model_path, map_location=device))
        else:
            print("No pretrained model found, starting from scratch")
        
        
        # # Check if an augmented model exists to start from
        # augmented_model_path = 'augmented_model.pth'
        # pretrained_model_path = 'pretrained_model.pth'
        
        # if os.path.exists(augmented_model_path):
        #     print(f"Loading previously augmented model from {augmented_model_path} as starting point")
        #     model.load_state_dict(torch.load(augmented_model_path, map_location=device))
        # elif os.path.exists(pretrained_model_path):
        #     print(f"Loading pretrained model from {pretrained_model_path} as starting point")
        #     model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        # else:
        #     print("No pretrained model found, starting from scratch")
        
        model.to(device)
        
        # Define loss function and optimizer
        criterion = {
            'stroke_count': nn.MSELoss(),
            'first_stroke': nn.CrossEntropyLoss()
        }
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Log parameters to MLflow
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_stroke_count": max_stroke_count,
            "num_stroke_types": num_stroke_types,
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "augmentation_factor": augmentation_factor,
            "device": str(device)
        })
        
        # Best model state
        best_val_loss = float('inf')
        best_model_state = None
        
        # Early stopping parameters
        patience = 20
        counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_stroke_count_error = 0.0
            train_first_stroke_correct = 0
            train_samples = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                # Move data to device
                images = batch['image'].to(device)
                stroke_counts = batch['stroke_count'].float().to(device)
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
                train_stroke_count_error += torch.sum(torch.abs(outputs['stroke_count'] - stroke_counts)).item()
                # _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
                # train_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
                
                _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
                train_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
                
                train_samples += images.size(0)
            
            # Calculate epoch statistics
            train_loss = train_loss / train_samples
            train_stroke_count_mae = train_stroke_count_error / train_samples
            train_first_stroke_acc = train_first_stroke_correct / train_samples
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_stroke_count_error = 0.0
            val_first_stroke_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    # Move data to device
                    images = batch['image'].to(device)
                    stroke_counts = batch['stroke_count'].float().to(device)
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
                    val_stroke_count_error += torch.sum(torch.abs(outputs['stroke_count'] - stroke_counts)).item()
                    # _, stroke_count_preds = torch.max(outputs['stroke_count'], 1)
                    # val_stroke_count_correct += torch.sum(stroke_count_preds == stroke_counts).item()
                    
                    _, first_stroke_preds = torch.max(outputs['first_stroke'], 1)
                    val_first_stroke_correct += torch.sum(first_stroke_preds == first_strokes).item()
                    
                    val_samples += images.size(0)
            
            # Calculate epoch statistics
            val_loss = val_loss / val_samples
            val_stroke_count_mae = val_stroke_count_error / val_samples
            val_first_stroke_acc = val_first_stroke_correct / val_samples
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Stroke Count MAE: {train_stroke_count_mae:.4f}, First Stroke Acc: {train_first_stroke_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Stroke Count MAE: {val_stroke_count_mae:.4f}, First Stroke Acc: {val_first_stroke_acc:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_stroke_count_mae": train_stroke_count_mae,
                "val_stroke_count_mae": val_stroke_count_mae,
                "train_first_stroke_acc": train_first_stroke_acc,
                "val_first_stroke_acc": val_first_stroke_acc
            }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(model, "best_model")
                
                # Save model locally
                torch.save(model.state_dict(), 'augmented_model.pth')
                
                # Reset early stopping counter
                counter = 0
            else:
                counter += 1
                
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save final model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f'augmented_model_final_{timestamp}.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to '{final_model_path}'")
        
        # Also save as the standard name for future runs
        torch.save(model.state_dict(), 'augmented_model.pth')
        print(f"Final model also saved to 'augmented_model.pth' for future runs")
        
        # Log final model as artifact
        mlflow.log_artifact(final_model_path)
        
        return model

if __name__ == "__main__":
    # Train model with augmentation
    train_model_with_augmentation(
        max_chars=7000,
        augmentation_factor=10,
        batch_size=64,
        num_epochs=200,
        learning_rate=0.0005  # Lower learning rate since we're starting from a pretrained model
    ) 