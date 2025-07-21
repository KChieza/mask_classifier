import os
import shutil
import random
import csv

# Paths
source_dir = "sey_beri_masks"
output_dir = "data"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
labels_file = os.path.join(output_dir, "labels.csv")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Prepare labels
labels = []

# Process the dataset
for label, subfolder in enumerate(["passed", "failed"]):
    subfolder_path = os.path.join(source_dir, subfolder)
    images = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle and split into train and validation sets
    random.shuffle(images)
    split_idx = int(0.8 * len(images))  # 80% train, 20% validation
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Move images and record labels
    for img in train_images:
        shutil.copy(os.path.join(subfolder_path, img), os.path.join(train_dir, img))
        labels.append((img, label))
    
    for img in val_images:
        shutil.copy(os.path.join(subfolder_path, img), os.path.join(val_dir, img))
        labels.append((img, label))

# Write labels.csv
with open(labels_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])  # Header
    writer.writerows(labels)

print("Dataset conversion complete!")