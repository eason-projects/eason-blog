import torch
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from pretrain_cnn import StrokeOrderPretrainDataset

# Import the augmentation class from the new script
from pretrain_cnn_augmented import AugmentedStrokeOrderDataset

def visualize_augmentations(num_samples=5, augmentation_factor=8):
    """Visualize augmentations for a few samples"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create base dataset
    base_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        max_chars=7000,
        train=True,
        split_ratio=0.9
    )
    
    # Create augmented dataset
    augmented_dataset = AugmentedStrokeOrderDataset(
        base_dataset=base_dataset,
        augmentation_factor=augmentation_factor
    )
    
    print(f"Base dataset size: {len(base_dataset)}")
    print(f"Augmented dataset size: {len(augmented_dataset)}")
    
    # Select random samples
    indices = random.sample(range(len(base_dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, augmentation_factor, figsize=(15, 3 * num_samples))
    
    for i, base_idx in enumerate(indices):
        # Get original sample
        base_sample = base_dataset[base_idx]
        char = base_sample['character']
        stroke_count = base_sample['stroke_count']
        
        # Display original and augmented versions
        for j in range(augmentation_factor):
            # Calculate index in augmented dataset
            aug_idx = base_idx * augmentation_factor + j
            
            # Get augmented sample
            aug_sample = augmented_dataset[aug_idx]
            
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
                title = f"{char} ({augmented_dataset.augmentation_types[j]})"
            
            # Use grayscale colormap for all images
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    print("Augmentation examples saved to 'augmentation_examples.png'")
    plt.show()

if __name__ == "__main__":
    visualize_augmentations(num_samples=5, augmentation_factor=10) 