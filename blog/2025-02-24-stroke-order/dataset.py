import json
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from models import MobileNetV3Model

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

# Custom feature extractor using the pretrained CNN model
class PretrainedCombinedExtractor(nn.Module):
    def __init__(self, observation_space, pretrained_model_path='augmented_model.pth'):
        super().__init__()
        
        # Load the pretrained CNN model
        # Get number of stroke types and max stroke count from the environment
        max_stroke_count = 36
        num_stroke_types = 27
        
        # Create the pretrained model
        self.pretrained_model = MobileNetV3Model(max_stroke_count, num_stroke_types)
        
        # Try to load the pretrained weights
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Successfully loaded pretrained model from {pretrained_model_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Using randomly initialized weights instead")
        
        # Extract only the CNN backbone from the pretrained model
        self.features = self.pretrained_model.features
        
        # Add a global average pooling layer to ensure flat output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze the CNN weights to use as a feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Get the CNN output size by doing a forward pass with a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)
            features_output = self.features(dummy_input)
            pooled_output = self.global_pool(features_output)
            cnn_output_size = pooled_output.reshape(1, -1).shape[1]
        
        # Linear layer for processing stroke history
        stroke_history_size = observation_space['stroke_history'].shape[0]
        self.stroke_encoder = nn.Sequential(
            nn.Linear(stroke_history_size, 128),
            nn.ReLU()
        )
        
        # Combine features
        self.combined = nn.Sequential(
            nn.Linear(cnn_output_size + 128, 512),
            nn.ReLU()
        )
        
        # Set the features dimension for stable-baselines3
        self.features_dim = 512
        
    def forward(self, observations):
        # Process image with pretrained CNN
        x = self.features(observations['image'].float())
        
        # Apply global average pooling and flatten
        x = self.global_pool(x)
        image_features = x.reshape(x.size(0), -1)
        
        # Process stroke history
        stroke_features = self.stroke_encoder(observations['stroke_history'].float())
        
        # Combine features
        combined_features = torch.cat([image_features, stroke_features], dim=1)
        return self.combined(combined_features)
