import os
import random
import shutil
import pandas as pd
import cv2
from glob import glob

# Files and Directories
IMG_DIR = 'images'
LABEL_DIR = 'labels'
TRAIN_IMG_DIR = 'targets/images/train/'
VAL_IMG_DIR = 'targets/images/val/'
TRAIN_LABEL_DIR = 'targets/labels/train/'
VAL_LABEL_DIR = 'targets/labels/val/'
CSV_PTH = 'labels_mnetv2.csv'

# Constants (px)
BOX_WIDTH = 64
BOX_HEIGHT = 64

# Sanity check function
def sanity_check(img_dir, label_dir, split_name):
    img_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
    label_files = sorted(glob(os.path.join(label_dir, '*.txt')))
    
    img_names = set(os.path.splitext(os.path.basename(f))[0] for f in img_files)
    label_names = set(os.path.splitext(os.path.basename(f))[0] for f in label_files)

    missing_labels = img_names - label_names
    missing_images = label_names - img_names

    print(f"\n--- {split_name.upper()} SET CHECK ---")
    print(f"Images: {len(img_files)}, Labels: {len(label_files)}")

    if missing_labels:
        print(f"⚠️ Missing labels for: {sorted(missing_labels)}")
    if missing_images:
        print(f"⚠️ Missing images for: {sorted(missing_images)}")
    if not missing_labels and not missing_images:
        print("✅ All images and labels match.")

# Create directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PTH, header=None, names=["filename", "x", "y", "class"])

# Group by image
for img_name, group in df.groupby("filename"):
    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # Get image dimensions
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    box_w = BOX_WIDTH / w
    box_h = BOX_HEIGHT / h

    with open(label_path, 'w') as f:
        for _, row in group.iterrows():
            # Assuming x, y are bbox centers, and width/height are fixed or estimated
            x_center = float(row['x']) / w
            y_center = float(row['y']) / h
            f.write(f"{row['class']} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

# Config shuffle
# Get only valid image files
all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle randomly
random.shuffle(all_images)

# Split
split_ratio = 0.8 # 80% of dataset for training
split_idx = int(len(all_images) * split_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Copy Images and Labels
for img in train_images:
    shutil.copy(os.path.join(IMG_DIR, img), os.path.join(TRAIN_IMG_DIR, img))
    label_name = os.path.splitext(img)[0] + ".txt"
    shutil.copy(os.path.join(LABEL_DIR, label_name), os.path.join(TRAIN_LABEL_DIR, label_name))

for img in val_images:
    shutil.copy(os.path.join(IMG_DIR, img), os.path.join(VAL_IMG_DIR, img))
    label_name = os.path.splitext(img)[0] + ".txt"
    shutil.copy(os.path.join(LABEL_DIR, label_name), os.path.join(VAL_LABEL_DIR, label_name))

# Classes
class_map = {
    0: 'alien',
    1: 'john'
}

# Generate data.yaml from classes
id_to_name = [class_map[i] for i in sorted(class_map)]

with open('data.yaml', 'w') as f:
    f.write(f"train: datasets/{TRAIN_IMG_DIR}\n")
    f.write(f"val: datasets/{VAL_IMG_DIR}\n")
    f.write(f"nc: {len(class_map)}\n")
    f.write(f"names: {id_to_name}\n")

# Run checks on both splits
sanity_check(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, "train")
sanity_check(VAL_IMG_DIR, VAL_LABEL_DIR, "val")
