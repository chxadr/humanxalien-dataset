# Adrien CHEVRIER MIT-0 License: LICENSES/MIT-0.txt

"""
mnetv2_burst_labeling.py

Captures images from a camera for dataset creation, saving cropped
frames and labeling them automatically in a CSV file, using
constant object's center coordinates inside the label of
each capture. For example, you can use it to take background pictures.

This script performs the following operations:

- Captures frames at specified resolution and framerate.
- Crops each frame to a fixed size and position.
- Saves captured frames in a `<class_name>/` directory.
- Logs filename, constants center coordinates, and label in a CSV file,
  with the following layout:
  
    <class_name>_<n>.jpg,<center_x>,<center_y>,<class_id>

Adapt this script based on your model training requirements.
"""

import cv2
import os
import csv
import time

# Classes
CLASSES = [ "alien", "john", "background" ]
TARGET = "background"

# Camera settings (320x240 MJPG @ 120 FPS)
CAP_WIDTH = 320
CAP_HEIGHT = 240
CAP_FPS = 120
CAP_MODE = "MJPG"

# Capture settings
NUM_IMAGES = 500                                     # Number of frames to capture
CAPTURE_INTERVAL = 1.0 // (CAP_FPS // 2)             # Time between each capture [s] (must be >= 1 / CAP_FPS)
CROP_SIZE = 224                                      # Frame size for dataset [px]
CROP_X, CROP_Y = 0, 0                                # Top-left corner reference position for crop [px,px]
CENTER_X, CENTER_Y = CROP_SIZE // 2, CROP_SIZE // 2  # Constant position set for the captured instance [px,px]

# CSV file to link captured frames filenames with labels
CSV_PTH = "labels.csv"

# Frame counter
count = 0

# Open /dev/video0
cap = cv2.VideoCapture(0)

# Camera initialization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAP_MODE))

if not cap.isOpened():
    print("[Error] Cannot open video device. Abort main program")
    exit()

# Create dataset directory and CSV file if needed
os.makedirs("captures", exist_ok=True)
if not os.path.exists(CSV_PTH):
    with open(CSV_PTH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", 'x', 'y', "label"])

print(f"[Info] Capturing {NUM_IMAGES} images ({CAPTURE_INTERVAL} s interval) ...")

while count < NUM_IMAGES:
    
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("[Error] Failed to grab frame. Retrying ...")
        continue

    # Crop frame to CROP_SIZExCROP_SIZE starting from (CROP_X, CROP_Y) top-left corner
    cropped_frame = frame[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]

    # Save frame
    filename = f"captures/{TARGET}_{count}.jpg"
    cv2.imwrite(filename, cropped_frame)
    print(f"[Info] [{count+1}/{NUM_IMAGES}] saved: {filename}")

    # Appends frame filename and label to CSV
    with open(CSV_PTH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(filename), CENTER_X, CENTER_Y, CLASSES.index(TARGET)])

    # Increments frame counter and wait for next capture
    count += 1
    time.sleep(CAPTURE_INTERVAL)

# Release memory and exit
cap.release()
cv2.destroyAllWindows()
print("[Info] Image capture completed")
