import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_numeric_id(filename):
    """
    Extracts the numeric ID from a filename.
    Example: RGB10132.jpg -> 10132, thermal0001.jpg -> 0001
    """
    match = re.search(r'\d+', filename)
    return match.group() if match else None


class ValidDataset(Dataset):
    """
    Dataset for validation without augmentations or cropping.

    Parameters:
    rgb_dir (str): Directory containing RGB images.
    thermal_dir (str): Directory containing thermal images.
    transform (callable, optional): Transformations to apply to the images.
    """
    def __init__(self, rgb_dir, thermal_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform

        # List files in directories
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.thermal_images = sorted(os.listdir(thermal_dir))

        # Match files by numeric ID
        rgb_ids = {extract_numeric_id(f): f for f in self.rgb_images}
        thermal_ids = {extract_numeric_id(f): f for f in self.thermal_images}
        common_ids = set(rgb_ids.keys()).intersection(thermal_ids.keys())

        # Create valid pairs
        self.image_pairs = [
            (os.path.join(rgb_dir, rgb_ids[id]), os.path.join(thermal_dir, thermal_ids[id]))
            for id in common_ids
        ]

        if len(self.image_pairs) == 0:
            raise ValueError("No matching RGB and thermal image pairs found. Check your data directories.")

    def __getitem__(self, index):
        rgb_path, thermal_path = self.image_pairs[index]

        # Open images
        rgb = Image.open(rgb_path).convert('RGB')
        thermal = Image.open(thermal_path).convert('L')  # Thermal is single channel

        # Apply transformations if provided
        if self.transform:
            rgb, thermal = self.transform(rgb, thermal)

        return rgb, thermal

    def __len__(self):
        return len(self.image_pairs)


class RandomTrainDataset(Dataset):
    """
    Dataset for training without cropping and augmentations.

    Parameters:
    rgb_dir (str): Directory containing RGB images.
    thermal_dir (str): Directory containing thermal images.
    transform (callable, optional): Transformations to apply to the images.
    """
    def __init__(self, rgb_dir, thermal_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform

        # List files in directories
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.thermal_images = sorted(os.listdir(thermal_dir))

        # Match files by numeric ID
        rgb_ids = {extract_numeric_id(f): f for f in self.rgb_images}
        thermal_ids = {extract_numeric_id(f): f for f in self.thermal_images}
        common_ids = set(rgb_ids.keys()).intersection(thermal_ids.keys())

        # Create valid pairs
        self.image_pairs = [
            (os.path.join(rgb_dir, rgb_ids[id]), os.path.join(thermal_dir, thermal_ids[id]))
            for id in common_ids
        ]

        if len(self.image_pairs) == 0:
            raise ValueError("No matching RGB and thermal image pairs found. Check your data directories.")

    def __getitem__(self, index):
        rgb_path, thermal_path = self.image_pairs[index]

        # Convert images to numpy arrays normalized to [0, 1]
        rgb = np.array(Image.open(rgb_path)) / 255.0
        thermal = np.array(Image.open(thermal_path)) / 255.0

        # Ensure the arrays are contiguous
        rgb = rgb.copy()
        thermal = thermal.copy()

        # Convert to channel-first format
        rgb = np.transpose(rgb, (2, 0, 1))
        thermal = np.expand_dims(thermal, axis=0)

        # Apply transformations if provided
        if self.transform:
            rgb, thermal = self.transform(rgb, thermal)

        # Convert to PyTorch tensors
        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(thermal, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.image_pairs)

