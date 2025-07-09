# File to perform capture and lebeling frame by frame.

import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Camera settings
CAP_WIDTH = 320
CAP_HEIGHT = 240
CAP_FPS = 120
CAP_MODE = "MJPG"

# Label and save parameters
csv_file = 'labels.csv'
last_label_param = 1 # 1 for John, 0 for alien.

# Frame counter
count = 0

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAP_MODE))

if not cap.isOpened():
    print("Error: Cannot open video device.")
    exit()

os.makedirs('captures', exist_ok=True)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'x', 'y'])

frame = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)  # Black image
last_click_coords = None

window_name = 'Video Playback'
cv2.imshow(window_name, frame)

print(f"Camera resolution set to: {int(CAP_WIDTH)}x{int(CAP_HEIGHT)}@{int(CAP_FPS)}fps with {CAP_MODE}")
print("Click on the window to capture a new frame and set label coordinates.")
print("Press SPACE to save the displayed frame with coordinates in CSV.")
print("Press 'q' to quit.")

def on_mouse(event, x, y, flags, param):
    global frame, last_click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        # Flush the buffer
        for _ in range(5):
            cap.read()

        ret, new_frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            return

        last_click_coords = (x, y)
        frame = new_frame  # Keep raw frame for saving

        # Make a preview copy with a 224x224 square from top-left
        preview_frame = new_frame.copy()
        cv2.rectangle(preview_frame, (0, 0), (224, 224), (0, 255, 0), 2)
        cv2.imshow(window_name, preview_frame)

        print(f"Selected fresh frame. Label: {last_click_coords}")

cv2.setMouseCallback(window_name, on_mouse)

while True:

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break

    elif key == 32:  # Space bar
        
        if frame is None:
            print("No frame captured yet. Click on the window first.")
            continue

        if last_click_coords is None:
            print("No label coordinates set. Click on the window to set label first.")
            continue

        # Crop 224x224 from top-left
        cropped_frame = frame[0:224, 0:224]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'captures/capture_{timestamp}.jpg'
        cv2.imwrite(filename, cropped_frame)
        print(f"Saved cropped frame to {filename}")

        x, y = last_click_coords
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(filename), x, y, last_label_param])
            count += 1
        print(f"Saved filename and coordinates to {csv_file}")
        print(f"Saved: {count}")

cap.release()
cv2.destroyAllWindows()
