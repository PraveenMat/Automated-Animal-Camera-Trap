import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd

# CSV file with labels
label_data = pd.read_csv('C:\\Users\\prm24\\OneDrive - University of Sussex\\Masters\\FYP\\Label Data\\Train Data.csv')

# Base directories
video_base_dir = 'C:\\Users\\prm24\\OneDrive - University of Sussex\\Masters\\FYP\\Camera Data\\Train Data'
output_dir = 'C:\\Users\\prm24\\OneDrive - University of Sussex\\Masters\\FYP\\Submission\\Code - Submission\\Frames'

# Ensure output dir exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each row in the label data
for index, row in label_data.iterrows():
    # Split the Video column to get the folder name and file name
    video_path_parts = row['Video'].strip().split('/')
    folder_name = video_path_parts[0].strip()  # one example 'N_N 3_2020_02_07'
    file_name = video_path_parts[1].strip()  # 'IMG_0002.MP4'
        
    subject = row['Subject']
    
    # Handles missing or NaN in Subjects
    if pd.isna(subject):
        print(f"Missing subject in row {index}. Skiped")
        continue

    subject = subject.strip()  # Example: 'LONGHORN CATTLE'
    
    # Construct the full path to the video file
    video_path = os.path.join(video_base_dir, folder_name, file_name)
    
    # Print the video path
    print(f"Video path: {video_path}")
    
    # Check if the file exists
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        continue
    
    # Extract frames every 5 seconds (0, 5, 10, 15, 20, 25, 30 seconds)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    for timestamp in range(0, 31, 5):  # range from 0 to 30 with step of 5
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Move to specific second

        ret, frame = cap.read()
        if ret:
            # Updates the naming convention for the frame
            frame_name = f"{folder_name}-{file_name.split('.')[0]}-{timestamp}.jpg"
            class_folder = os.path.join(output_dir, subject)
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            frame_path = os.path.join(class_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            print(f"Extracted the frame at {timestamp} seconds from {file_name} in {folder_name}")
        else:
            print(f"Failed extraction of the frame at {timestamp} seconds from {file_name} in {folder_name}")

    cap.release()

print("Frame Extraction Completed")

# List all images and their labels
all_images = []
all_labels = []

# Iterates through all subdir
for label in os.listdir(output_dir):
    class_folder = os.path.join(output_dir, label)
    images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
    all_images.extend(images)
    all_labels.extend([label] * len(images))

# Error check if there are any images found
if not all_images:
    print("ERROR, No images was extracted. Check the path again")
else:
    # Splits data into training and validation set (80% train, 20% val)
    train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, stratify=all_labels)

    # Create dir for train and val data
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for label in set(all_labels):
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)

    # Move files to train and val directories
    for img, label in zip(train_images, train_labels):
        shutil.move(img, os.path.join(train_dir, label))

    for img, label in zip(val_images, val_labels):
        shutil.move(img, os.path.join(val_dir, label))

    print("Data is split into training and validation set")
