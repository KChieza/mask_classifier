from torch.utils.data import Dataset
from PIL import Image
import os

class FishMaskDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = self._load_samples(label_file)

    def _load_samples(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        samples = []
        for line in lines:
            path, label = line.strip().split(',')  # e.g., "mask1.png,1"
            samples.append((path, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_filename, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        
        # Load image as grayscale mask
        image = Image.open(img_path).convert('L')  # 1-channel (0-255)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
