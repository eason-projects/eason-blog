import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import the necessary classes from pretrain_cnn.py and pretrain_cnn_augmented.py
from pretrain_cnn import MobileNetV3Model, StrokeOrderPretrainDataset, CNNModel
from pretrain_cnn_augmented import MobileNetV3LiteModel

def evaluate_model(model_path, num_samples=20, random_samples=True, display_images=True, model_type="mobilenetv3lite"):
    """
    Evaluate the pretrained CNN model and print predictions for characters.
    
    Args:
        model_path: Path to the pretrained model
        num_samples: Number of samples to evaluate
        random_samples: If True, select random samples; otherwise, use the first num_samples
        display_images: If True, display the character images
        model_type: Type of model to use ("cnn", "mobilenetv3", or "mobilenetv3lite")
    """
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for evaluation")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    
    # Create validation dataset
    val_dataset = StrokeOrderPretrainDataset(
        stroke_order_path='./stroke-order-jian.json',
        stroke_table_path='./stroke-table.json',
        image_folder='./images',
        max_chars=7000,
        train=True,
        split_ratio=0.9
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Get number of stroke types from dataset
    dataset_max_stroke_count = max(sample['stroke_count'] for sample in val_dataset.samples)
    num_stroke_types = len(val_dataset.dataset.stroke_names)
    
    print(f"Dataset max stroke count: {dataset_max_stroke_count}")
    print(f"Number of stroke types: {num_stroke_types}")
    
    # Use the model's expected dimensions (from the error message)
    # The error showed the model expects 37 outputs for stroke count
    model_max_stroke_count = 36  # 37 - 1 (0-indexed)
    
    print(f"Using model max stroke count: {model_max_stroke_count}")
    
    # Create model with the correct dimensions based on model_type
    if model_type == "cnn":
        model = CNNModel(model_max_stroke_count, num_stroke_types)
        is_regression = False
    elif model_type == "mobilenetv3":
        model = MobileNetV3Model(model_max_stroke_count, num_stroke_types)
        is_regression = True
    elif model_type == "mobilenetv3lite":
        model = MobileNetV3LiteModel(model_max_stroke_count, num_stroke_types)
        is_regression = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load pretrained model
    print(f"Loading pretrained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Select samples to evaluate
    if random_samples:
        indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    else:
        indices = list(range(min(num_samples, len(val_dataset))))
    
    # Evaluate selected samples
    print("\nEvaluating model on selected samples:")
    print("-" * 80)
    print(f"{'Character':<10} {'Actual Count':<15} {'Predicted Count':<20} {'Actual First':<20} {'Predicted First':<20} {'Correct?':<10}")
    print("-" * 80)
    
    correct_count = 0
    correct_first = 0
    total_count_error = 0.0
    
    # Store results for display
    results = []
    
    with torch.no_grad():
        for idx in indices:
            sample = val_dataset[idx]
            
            # Get character, actual stroke count, and actual first stroke
            char = sample['character']
            actual_count = sample['stroke_count']
            actual_first = sample['first_stroke']
            
            # Get image and move to device
            image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass
            outputs = model(image)
            
            # Get predicted stroke count and first stroke
            if is_regression:
                # For regression model (MobileNetV3Lite)
                pred_count_raw = outputs['stroke_count'].item()
                pred_count = round(pred_count_raw)  # Round to nearest integer
                count_error = abs(pred_count - actual_count)
                total_count_error += count_error
                count_correct = count_error < 1  # Consider correct if error < 1
            else:
                # For classification models (CNN, MobileNetV3)
                _, pred_count = torch.max(outputs['stroke_count'], 1)
                pred_count = pred_count.item()
                count_correct = pred_count == actual_count
            
            # Get predicted first stroke (same for all models)
            _, pred_first = torch.max(outputs['first_stroke'], 1)
            pred_first = pred_first.item()
            
            # Get stroke names
            actual_first_name = val_dataset.dataset.stroke_names[actual_first]
            pred_first_name = val_dataset.dataset.stroke_names[pred_first]
            
            # Check if predictions are correct
            first_correct = pred_first == actual_first
            
            if count_correct:
                correct_count += 1
            if first_correct:
                correct_first += 1
            
            # Print results
            correct_mark = "✓" if count_correct and first_correct else "✗"
            print(f"{char:<10} {actual_count:<15} {pred_count:<20} {actual_first_name:<20} {pred_first_name:<20} {correct_mark:<10}")
            
            # Store results for display
            results.append({
                'char': char,
                'image': sample['image'].numpy(),
                'actual_count': actual_count,
                'pred_count': pred_count,
                'actual_first': actual_first_name,
                'pred_first': pred_first_name,
                'count_correct': count_correct,
                'first_correct': first_correct
            })
    
    # Print overall accuracy
    print("-" * 80)
    accuracy_count = correct_count / len(indices) * 100
    accuracy_first = correct_first / len(indices) * 100
    
    if is_regression:
        avg_count_error = total_count_error / len(indices)
        print(f"Stroke Count Accuracy: {accuracy_count:.2f}% ({correct_count}/{len(indices)})")
        print(f"Average Stroke Count Error: {avg_count_error:.2f}")
    else:
        print(f"Stroke Count Accuracy: {accuracy_count:.2f}% ({correct_count}/{len(indices)})")
    
    print(f"First Stroke Accuracy: {accuracy_first:.2f}% ({correct_first}/{len(indices)})")
    
    # Display images if requested
    if display_images:
        display_character_images(results)

def display_character_images(results):
    """Display character images with predictions"""
    
    # Calculate grid dimensions
    n_samples = len(results)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(15, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # Plot each character
    for i, result in enumerate(results):
        ax = plt.subplot(gs[i])
        
        # Display image
        img = result['image'].squeeze()  # Remove channel dimension
        ax.imshow(img, cmap='gray')
        
        # Instead of trying to display the Chinese character directly in the title,
        # we'll use the character image itself and add text annotations for predictions
        
        # Create a simple title without the Chinese character
        title = f"Strokes: {result['actual_count']}"
        if result['count_correct']:
            title += f" ✓"
        else:
            title += f" → {result['pred_count']} ✗"
            
        title += f"\nFirst: {result['actual_first']}"
        if result['first_correct']:
            title += f" ✓"
        else:
            title += f" → {result['pred_first']} ✗"
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig('character_evaluation.png', bbox_inches='tight')
        print(f"Character images saved to 'character_evaluation.png'")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    plt.show()

if __name__ == "__main__":
    model_path = './augmented_model_final_20250302_145758.pth'
    # Specify the model type based on the model you're evaluating
    # Options: "cnn", "mobilenetv3", or "mobilenetv3lite"
    evaluate_model(
        model_path, 
        num_samples=100, 
        random_samples=True, 
        display_images=True,
        model_type="mobilenetv3"  # Change this based on your model
    ) 