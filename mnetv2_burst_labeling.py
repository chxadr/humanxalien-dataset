import cv2
import os
import csv
import time

# Camera settings
CAP_WIDTH = 320
CAP_HEIGHT = 240
CAP_FPS = 120
CAP_MODE = "MJPG"

# Capture settings
NUM_IMAGES = 500                                   # Number of images to capture
CAPTURE_INTERVAL = 1.0 // (CAP_FPS // 2)           # Seconds between each capture
CROP_SIZE = 224
CROP_X, CROP_Y = 0, 0
LABEL_X, LABEL_Y = CROP_SIZE // 2, CROP_SIZE // 2  # Center of crop

# File settings
csv_file = 'labels.csv'
label_value = 2  # 2 for none, 1 for John, 0 for alien

# Frame counter
count = 0

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAP_MODE))

if not cap.isOpened():
    print("Error: Cannot open video device.")
    exit()

# Create directories and CSV if needed
os.makedirs('captures', exist_ok=True)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x', 'y', 'label'])

print(f"Capturing {NUM_IMAGES} images (1 per second)...")

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        continue

    # Crop 224x224 from top-left
    cropped_frame = frame[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]

    # Save image
    filename = f'captures/capture_{count}.jpg'
    cv2.imwrite(filename, cropped_frame)
    print(f"[{count+1}/{NUM_IMAGES}] Saved: {filename}")

    # Append label to CSV
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(filename), LABEL_X, LABEL_Y, label_value])

    count += 1
    time.sleep(CAPTURE_INTERVAL)

cap.release()
cv2.destroyAllWindows()
print("Image capture complete.")
