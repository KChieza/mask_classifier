from torchvision import transforms
from dataset import FishMaskDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = FishMaskDataset(
    image_dir='data/train',
    label_file='data/labels/train_labels.csv',
    transform=transforms
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

# Show 1 batch
data_iter = iter(dataloader)
images, labels = next(data_iter)

print("Image batch shape:", images.shape)  # [B, C, H, W]
print("Label batch:", labels)