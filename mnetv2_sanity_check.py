# Check if the image capture and labeling processes
# went well. Provide detailed information and
# user interactions.

import os
import csv
from collections import Counter, defaultdict
import cv2

# Paths
csv_file = 'labels.csv'
images_folder = 'captures/'

# Get all image files in folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
image_set = set(image_files)
image_count = len(image_files)

# Read all image names from CSV
csv_filenames = []
with open(csv_file, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            csv_filenames.append(row[0])
csv_set = set(csv_filenames)
csv_count = len(csv_filenames)

# Detect missing and orphaned files
missing_files = csv_set - image_set           # In CSV but not in folder
orphaned_files = image_set - csv_set          # In folder but not in CSV

# Detect duplicates
filename_counts = Counter(csv_filenames)
duplicates = {name: count for name, count in filename_counts.items() if count > 1}

# Group rows by filename to get coordinates
duplicate_entries = defaultdict(list)
with open(csv_file, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            filename = row[0]
            coords = row[1:3]  # x, y
            duplicate_entries[filename].append(coords)

# Print summary
print(f"🖼️  Image files in '{images_folder}': {image_count}")
print(f"📝 Entries in '{csv_file}': {csv_count}")

if missing_files:
    print("\n⚠️  Missing files referenced in CSV but not found in folder:")
    for f in sorted(missing_files):
        print(" -", f)
else:
    print("\n✅ No missing image files.")

if orphaned_files:
    print("\n🧹 Orphaned image files not listed in CSV:")
    for f in sorted(orphaned_files):
        print(" -", f)
else:
    print("\n✅ No orphaned image files.")

print("\n🔁 Duplicate entries with coordinates:")
found_duplicates = False
for filename, coords_list in duplicate_entries.items():
    if len(coords_list) > 1:
        found_duplicates = True
        print(f" - {filename} appears {len(coords_list)} times:")
        for coords in coords_list:
            print(f"    ↳ x={coords[0]}, y={coords[1]}")

if not found_duplicates:
    print("✅ No duplicate entries found in CSV.")

# Display duplicate images
print("\nDisplaying duplicated images. Press any key to show next, 'q' to quit.")

for filename, coords_list in duplicate_entries.items():
    if len(coords_list) > 1:
        image_path = os.path.join(images_folder, filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image {filename}")
                continue
            cv2.imshow(f"Duplicate: {filename}", img)
            print(f"Showing {filename} with {len(coords_list)} duplicates. Press any key to continue or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('q'):
                print("Quitting display loop.")
                break
        else:
            print(f"Image file not found: {filename}")
