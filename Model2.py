import os
import cv2
import csv
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import urllib.request
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Downloads Kinetics-400 labels. This from GitHub
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"

# Loads the Kinetics labels
def load_action_labels():
    with urllib.request.urlopen(KINETICS_URL) as obj:
        labels = [line.decode("utf-8").strip() for line in obj.readlines()]
    print("Found %d labels." % len(labels))
    return labels

# Kinetics labels
kinetics_labels = load_action_labels()

# Loads the YOLOv8 (YOLOv8 x varient) and SlowFast (slowfast_r101 varient) model
yolo_model = YOLO('yolov8x.pt').to(device)
slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True).eval().to(device)

# Converting video frames as a list of tensors for SlowFast
class PackPathway(torch.nn.Module):
    def __init__(self, slowfast_alpha=4):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

# YOLOv8 class for object detection and tracking using YOLOv8 and SlowFast
class YOLODetection:
    def __init__(self, capture, processed_video_dir, yolo_model, slowfast_model, detection_results):
        self.capture = capture
        self.processed_video_dir = processed_video_dir
        self.yolo_model = yolo_model
        self.slowfast_model = slowfast_model
        self.CLASS_NAMES_DICT = self.yolo_model.model.names
        self.detection_results = detection_results
        self.frame_buffers = defaultdict(list)

    # Performs object detection on the video using YOLOv8
    def predict(self, img):
        results = self.yolo_model(img, stream=True)
        return results

    # Plots the bounding boxes on the video using YOLOv8
    def plot_boundingBox(self, results, img):
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5:
                    detections.append(([x1, y1, w, h], conf, cls))

        return detections, img

    # Updates the tracker and performs behaviour recognition using SlowFast 
    def tracker_detecter(self, detections, img, folder, file_name, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            label = self.CLASS_NAMES_DICT.get(track.det_class, "Unknown")
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            matched_detection = None
            for detection in detections:
                (dx1, dy1, dw, dh), conf, cls = detection
                if abs(dx1 - x1) <= 5 and abs(dy1 - y1) <= 5 and abs(dw - w) <= 5 and abs(dh - h) <= 5:
                    matched_detection = detection
                    break

            if matched_detection:
                conf = matched_detection[1]
                display_text = f'{label} {conf:.2f} ID: {track_id}'
            else:
                display_text = f'{label} ID: {track_id}'

            object_frame = img[y1:y2, x1:x2]
            if object_frame.size > 0:
                self.frame_buffers[track_id].append(object_frame)

                # Performs behaviour recognition when 32 frames are collected
                if len(self.frame_buffers[track_id]) >= 32:
                    video_frames = self.frame_buffers[track_id][:32]
                    video_tensor = self.preprocess_frames(video_frames)

                    with torch.no_grad():
                        predictions = self.slowfast_model(video_tensor)

                    probs = torch.nn.functional.softmax(predictions, dim=1)
                    top_prob, top_label_idx = probs.topk(1)

                    top_label_idx_int = top_label_idx.item()
                    top_label = kinetics_labels[top_label_idx_int] if top_label_idx_int < len(kinetics_labels) else "Unknown"

                    self.detection_results[f"{folder}/{file_name}"][track_id] = (label, top_label, top_prob.item(), 30)  # Display for 30 frames

                    self.frame_buffers[track_id] = []

            # Display if there is track_id
            if track_id in self.detection_results[f"{folder}/{file_name}"]:
                label, behaviour_label, prob, display_counter = self.detection_results[f"{folder}/{file_name}"][track_id]
                if display_counter > 0:
                    display_text += f' Behaviour: {behaviour_label} ({prob:.2f})'
                    self.detection_results[f"{folder}/{file_name}"][track_id] = (label, behaviour_label, prob, display_counter - 1)

            # Displays the text on the video
            cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return img

    def preprocess_frames(self, frames):
        # Parameters for transformation for SlowFast
        side_size = 256
        crop_size = 256
        num_frames = 32
        slowfast_alpha = 4  
        
        # Resizes each frame to be consistent size and convert to tensor
        frames_resized = [cv2.resize(frame, (side_size, side_size)) for frame in frames]
        
        # Converts the list of frames into 4D tensor into [T, C, H, W]
        video_tensor = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames_resized])  # Now [T, C, H, W]

        # Permute to [C, T, H, W] to match the expected input shape
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Now [C, T, H, W]
        
        # Apply the transformation 
        transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway(slowfast_alpha)  
            ]
        )
        
        # Apply the transformation
        video_tensor = transform(video_tensor)  # Transform expects [C, T, H, W]
        
        # PackPathway returns a list tensors, we need to unsqueeze each one
        video_tensor = [t.unsqueeze(0) for t in video_tensor]  # Add batch dimension: [1, C, T, H, W]

        # The model expects two pathways (slow and fast) so return a list
        return [vt.to(device) for vt in video_tensor]

    def process_video(self, video_path, output_path, folder, file_name):
        print(f"Processing video: {video_path}")
        self.detection_results[f"{folder}/{file_name}"] = {}  # Initialise for each video

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            self.detection_results[f"{folder}/{file_name}"]['Error'] = 'Error: Could not open video'
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Initialises DEEP SORT for tracking each animal for each video
        tracker = DeepSort(max_age=25, n_init=2, nms_max_overlap=1.1)

        detections_made = False  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = self.predict(frame)
            detections, img = self.plot_boundingBox(results, frame)

            if detections:
                detections_made = True

            # Updates tracker
            outputs = self.tracker_detecter(detections, img, folder, file_name, tracker)

            # Writes the frame with the detection boxes
            out.write(outputs)

            # Display the frame 
            cv2.imshow('Object Detection', outputs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not detections_made:
            # If there was no detections found then add 'NA'
            self.detection_results[f"{folder}/{file_name}"]['NA'] = 'NA'
            print(f"No detections in video: {video_path}")
        else:
            print(f"Detections made in video: {video_path}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Finished processing {video_path}")

# Runs the YOLOv8 
def run_model(base_dir, processed_video_dir):
    detection_results = defaultdict(dict)
    output_csv = os.path.join(processed_video_dir, 'YOLOv8_Results.csv')
    process_videos(base_dir, processed_video_dir, detection_results, output_csv)

# Processes all the videos
def process_videos(base_dir, processed_video_dir, detection_results, output_csv):
    print(f"Scanning base directory: {base_dir}")
    all_folders = []
    all_video_files = []
    processed_files = set()

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            all_folders.append(folder_path)
            print(f"Entering folder: {folder_path}")
            all_files = os.listdir(folder_path)
            print(f"Files in folder {folder_path}: {all_files}")
            video_files = [f for f in all_files if f.lower().endswith('.mp4')]
            if not video_files:
                print(f"No video files found in folder: {folder_path}")
            all_video_files.extend([os.path.join(folder, f) for f in video_files])  # Include folder in file path to match the orgianl CSV struture
            for file in video_files:
                video_path = os.path.join(folder_path, file)
                output_file_name = f"{folder}-{file}"
                output_path = os.path.join(processed_video_dir, output_file_name)
                print(f"Processing {video_path} -> {output_path}")
                detector = YOLODetection(video_path, processed_video_dir, yolo_model, slowfast_model, detection_results)
                detector.process_video(video_path, output_path, folder, file)
                processed_files.add(os.path.join(folder, file))  # Processed file path
                print(f"Processed: {file}")

    missing_files = set(all_video_files) - processed_files
    if missing_files:
        print(f"Missing files: {missing_files}")
        for missing_file in missing_files:
            detection_results[missing_file]['NA'] = 'NA'  

    # Saving YOLO results as CSV
    print(f"Saving predictions results to {output_csv}")
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Video', 'Subject', 'Animal ID', 'Behaviour', 'Count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for file, objects in detection_results.items():
            if 'NA' in objects:
                writer.writerow({'Video': file, 'Subject': 'NA', 'Animal ID': 'NA', 'Behaviour': 'NA', 'Count': 0})
            else:
                object_count = defaultdict(int)
                for track_id, values in objects.items():
                    label, action_label, prob, display_counter = values  # Unpack all 4 elements
                    object_count[label] += 1
                    writer.writerow({
                        'Video': file,
                        'Subject': label,
                        'Animal ID': track_id,
                        'Behaviour': action_label,
                        'Count': object_count[label]
                    })

    print("Processing complete. CSV results saved to", output_csv)