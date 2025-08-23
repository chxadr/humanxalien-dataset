# Adrien CHEVRIER MIT-0 License: LICENSES/MIT-0.txt

"""
mnetv2_manual_labeling.py

Captures images from a camera for dataset creation, saving cropped frames
and labeling them automatically in a CSV file, using a left mouse click to
set the object's center coordinates inside the label of each capture.

This script performs the following operations:

- Captures frames at specified resolution and framerate.
- Crops each frame to a fixed size and position.
- Saves captured frames in a `<class_name>/` directory.
- Logs filename, center coordinates selected by a left mouse click,
  and label in a CSV file, with the following layout:
  
    <class_name>_<n>.jpg,<center_x>,<center_y>,<class_id>

Adapt this script based on your model training requirements.
"""

import cv2
import numpy as np
import os
import csv

# Classes
CLASSES = [ "alien", "john", "background" ]
TARGET = "john"

# Camera settings (320x240 MJPG @ 120 FPS)
CAP_WIDTH = 320
CAP_HEIGHT = 240
CAP_FPS = 120
CAP_MODE = "MJPG"

# Capture settings
NUM_IMAGES = 500      # Number of frames to capture
CROP_SIZE = 224       # Frame size for dataset [px]
CROP_X, CROP_Y = 0, 0 # Top-left corner reference position for crop [px,px]

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
os.makedirs(f"{TARGET}", exist_ok=True)
if not os.path.exists(CSV_PTH):
    with open(CSV_PTH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", 'x', 'y'])

# Black placeholder image at start
frame = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
last_click_coords = None

# Prepare OpenCV window to display camera output
window_name = "Video Playback"
cv2.imshow(window_name, frame)

# Instructions for the user
print("[Info] Click on the window to capture a new frame and set label coordinates.")
print("[Info] Press SPACE to save the displayed frame with coordinates in CSV.")
print("[Info] Press 'q' to quit.")


def on_mouse(event, x, y, flags, param):
    """Mouse callback function to capture mouse events.
    
    On a left mouse click, update reference to the newest frame
    and store the mouse absolute position on the image in a variable.
    
    Args:
      event: Mouse event to capture.
      x: Column index [px].
      y: Row index [px].
      flags: Event modifiers, e.g. Shift or Alt keys (Not Needed).
      param: Additionnal data (Not Needed).
    """
    
    global frame, last_click_coords
    
    # On left mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Flush the camera buffer
        for _ in range(5):
            cap.read()
        
        # Capture a frame
        ret, new_frame = cap.read()
        if not ret:
            print("[Error] Failed to grab frame.")
            return
        
        # Store mouse click absolute coordintates ([px], [px])
        last_click_coords = (x, y)
        # Update global frame variable to point on the new frame
        frame = new_frame
        
        # Make a preview with a CROP_SIZExCROP_SIZE green square starting from (CROP_X, CROP_Y) top-left corner
        preview_frame = new_frame.copy()
        cv2.rectangle(preview_frame, (CROP_X, CROP_Y), (CROP_X+CROP_SIZE, CROP_Y+CROP_SIZE), (0, 255, 0), 2)
        cv2.imshow(window_name, preview_frame)

        print(f"[Info] Selected fresh frame. Label: {last_click_coords}")

cv2.setMouseCallback(window_name, on_mouse)

while count < NUM_IMAGES:

    key = cv2.waitKey(0) & 0xFF
    
    # Quit program with `q`
    if key == ord('q'):
        break

    # Save capture and label with `SPACE`
    elif key == 32:
        
        # Ignore placeholder image
        if frame is None:
            print("[Warning] No frame captured yet. Click on the window first.")
            continue

        # Ignore empty coordinates
        if last_click_coords is None:
            print("[Warning] No label coordinates set. Click on the window to set label first.")
            continue

        # Crop frame to CROP_SIZExCROP_SIZE starting from (CROP_X, CROP_Y) top-left corner
        cropped_frame = frame[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]

        # Save captured frame
        filename = f"{TARGET}/{TARGET}_{count}.jpg"
        cv2.imwrite(filename, cropped_frame)
        print(f"[Info] Saved cropped frame to {filename}")

        # Create the label with mouse click coordinates
        # and save it with its corresponding capture filename in the CSV file
        x, y = last_click_coords
        with open(CSV_PTH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(filename), x, y, CLASSES.index(TARGET)])
            count += 1
        print(f"[Info] Saved filename and coordinates to {CSV_PTH}")
        print(f"[Info] Saved: {count}")

# Release memory and exits
cap.release()
cv2.destroyAllWindows()
print("[Info] Image capture completed")
