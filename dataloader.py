import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob

# Custom dataset for loading thermal images
class ThermalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(data_dir + '/*.jpg')
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations and batch size
transform = transforms.Compose([
    transforms.ToTensor()

])
batch_size = 32

# Load dataset and create DataLoader
train_data = ThermalImageDataset(data_dir='images_thermal_train/data', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Sample display of batch processing
data_iter = iter(train_loader)
images = next(data_iter)
print(f"Batch size: {images.size()}")
