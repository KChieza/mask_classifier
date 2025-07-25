import os
import shutil
import random
import csv

# Paths
source_dir = "sey_beri_masks"
output_dir = "data_test"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
train_labels_file = os.path.join(output_dir, "train_labels.csv")
val_labels_file = os.path.join(output_dir, "val_labels.csv")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Prepare labels
train_labels = []
val_labels = []

# Predefined subset size (set to 0 for all samples)
subset_size = 50

# Global counter for unique filenames
global_counter = 1

# Process the dataset
for label, subfolder in enumerate(["passed", "failed"]):
    subfolder_path = os.path.join(source_dir, subfolder)
    images = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Apply subset size if specified
    if subset_size > 0:
        images = images[:subset_size]
    
    # Shuffle and split into train and validation sets
    random.shuffle(images)
    split_idx = int(0.8 * len(images))  # 80% train, 20% validation
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Move images and record labels
    for img in train_images:
        new_name = f"frame_{global_counter:05d}.png"
        shutil.copy(os.path.join(subfolder_path, img), os.path.join(train_dir, new_name))
        train_labels.append((new_name, label))
        global_counter += 1
    
    for img in val_images:
        new_name = f"frame_{global_counter:05d}.png"
        shutil.copy(os.path.join(subfolder_path, img), os.path.join(val_dir, new_name))
        val_labels.append((new_name, label))
        global_counter += 1

# Write train_labels.csv
with open(train_labels_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    # writer.writerow(["filename", "label"])  # Removed header
    writer.writerows(train_labels)

# Write val_labels.csv
with open(val_labels_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    # writer.writerow(["filename", "label"])  # Removed header
    writer.writerows(val_labels)

print("Dataset conversion complete!")