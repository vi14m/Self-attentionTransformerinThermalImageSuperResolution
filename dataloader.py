import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class RandomTrainDataset(Dataset):
    def __init__(self, crop_size, augment=True, dbg=False, upscale=8):
        """
        Dataset for training with random crops and optional augmentations.

        Parameters:
        crop_size (int): Size of the crops to extract from the images.
        augment (bool): Whether to apply data augmentation.
        dbg (bool): Debug mode flag.
        upscale (int): Upscaling factor (8 or 16) for the low-resolution thermal images.
        """
        self.HR_vis_dir = 'flir/images_rgb_train/data/'
        self.HR_thermal_dir = 'flir/images_thermal_train/data/'
        self.upscale = upscale
        self.augment = augment
        self.crop_size = crop_size
        self.dbg = dbg

        # List all files in HR thermal directory
        self.keys = sorted(self._get_image_files(self.HR_thermal_dir))

        self.hr_images = []
        self.rgb_images = []
        self.lr_images = []

        for key in self.keys:
            identifier = self.get_common_identifier(key)

            HR_thermal_path = os.path.join(self.HR_thermal_dir, key)
            HR_vis_path = None

            # Search for RGB images based on identifier
            for file in self._get_image_files(self.HR_vis_dir):
                if identifier in file:
                    HR_vis_path = os.path.join(self.HR_vis_dir, file)
                    break

            # Generate LR thermal image
            LR_thermal_path = f"{HR_thermal_path.replace('.png', '')}-LR.png"
            if not os.path.exists(LR_thermal_path):
                # Downsample HR thermal to create LR image
                HR_thermal = Image.open(HR_thermal_path)
                LR_thermal = HR_thermal.resize((HR_thermal.width // self.upscale, HR_thermal.height // self.upscale), Image.BICUBIC)
                LR_thermal.save(LR_thermal_path)  # Save the generated LR image

            # Load images
            hr = np.array(Image.open(HR_thermal_path)) / 255.0
            rgb = np.array(Image.open(HR_vis_path)) / 255.0
            lr = np.array(Image.open(LR_thermal_path)) / 255.0

            self.hr_images.append(hr)
            self.rgb_images.append(rgb)
            self.lr_images.append(lr)

    def _get_image_files(self, directory):
        """
        Get a list of image files in a directory, excluding non-image files.

        Parameters:
        directory (str): The directory to search for image files.

        Returns:
        list: List of valid image file names.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

    def get_common_identifier(self, filename):
        """
        Extract a common identifier from the filename.

        Parameters:
        filename (str): The filename of the image.

        Returns:
        str: The common part of the filename (identifier).
        """
        identifier = filename.split('-')[0]  # Modify based on your filename structure
        return identifier

    def __getitem__(self, idx):
        """
        Get a data sample for training with random crop and augmentation.

        Parameters:
        idx (int): Index of the sample.

        Returns:
        tuple: Low-resolution thermal, high-resolution visible, and high-resolution thermal images as tensors.
        """
        rgb = self.rgb_images[idx]
        hr = self.hr_images[idx]
        lr = self.lr_images[idx]

        h, w, _ = lr.shape
        xx = random.randint(0, h - self.crop_size)
        yy = random.randint(0, w - self.crop_size)

        crop_rgb = rgb[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale, yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale, :]
        crop_hr = hr[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale, yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale, 0]
        crop_lr = lr[xx:xx+self.crop_size, yy:yy+self.crop_size, 0]

        crop_rgb = np.transpose(crop_rgb, (2, 0, 1))
        crop_hr = np.expand_dims(crop_hr, axis=0)
        crop_lr = np.expand_dims(crop_lr, axis=0)

        if self.augment:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            crop_lr = self.augment_image(crop_lr, rotTimes, vFlip, hFlip)
            crop_rgb = self.augment_image(crop_rgb, rotTimes, vFlip, hFlip)
            crop_hr = self.augment_image(crop_hr, rotTimes, vFlip, hFlip)

        crop_lr = np.ascontiguousarray(crop_lr, dtype=np.float32)
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.float32)
        crop_hr = np.ascontiguousarray(crop_hr, dtype=np.float32)

        return torch.tensor(crop_lr), torch.tensor(crop_rgb), torch.tensor(crop_hr)

    def __len__(self):
        return len(self.rgb_images)

    def augment_image(self, img, rotTimes, vFlip, hFlip):
        """
        Apply random augmentation to an image.

        Parameters:
        img (np.array): Image to augment.
        rotTimes (int): Number of 90-degree rotations.
        vFlip (bool): Whether to flip vertically.
        hFlip (bool): Whether to flip horizontally.

        Returns:
        np.array: Augmented image.
        """
        for _ in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))
        if vFlip:
            img = img[:, :, ::-1]
        if hFlip:
            img = img[:, ::-1, :]
        return img

class ValidDataset(Dataset):
    def __init__(self, upscale=8):
        """
        Dataset for validation images.

        Parameters:
        upscale (int): Upscaling factor (8 or 16) for the low-resolution thermal images.
        """
        self.HR_vis_dir = 'flir/images_rgb_val/data/'
        self.HR_thermal_dir = 'flir/images_thermal_val/data/'
        self.upscale = upscale

        # Assuming all files in HR thermal directory should have corresponding pairs
        self.keys = sorted(self._get_image_files(self.HR_thermal_dir))

    def _get_image_files(self, directory):
        """
        Get a list of image files in a directory, excluding non-image files.

        Parameters:
        directory (str): The directory to search for image files.

        Returns:
        list: List of valid image file names.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png')
        return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

    def __getitem__(self, index):
        """
        Get a data sample for validation.

        Parameters:
        index (int): Index of the sample.

        Returns:
        tuple: Low-resolution thermal, high-resolution visible, and high-resolution thermal images as tensors.
        """
        key = self.keys[index]

        # Assuming HR thermal and HR visible share the same filename in different directories
        HR_thermal_path = os.path.join(self.HR_thermal_dir, key)
        HR_vis_path = os.path.join(self.HR_vis_dir, key)

        HR_vis = np.array(Image.open(HR_vis_path)) / 255.0
        HR_thermal = np.expand_dims(np.array(Image.open(HR_thermal_path))[:, :, 0] / 255.0, axis=0)

        HR_vis = np.transpose(HR_vis, (2, 0, 1))  # Convert RGB image to channel-first format

        return torch.tensor(HR_thermal, dtype=torch.float32), \
               torch.tensor(HR_vis, dtype=torch.float32)

    def __len__(self):
        return len(self.keys)
