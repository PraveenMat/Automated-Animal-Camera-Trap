import os
import cv2
import csv
import torch
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from collections import defaultdict

# Load the trained VGG16 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=False)
vgg16.classifier[6] = torch.nn.Linear(vgg16.classifier[6].in_features, 14)  # 14 classes from the extracted frames from the training video data
vgg16.load_state_dict(torch.load('vgg16Custom.pth'))
vgg16 = vgg16.to(device).eval()

# Class names
class_names = [
    "BIRD", "DOG", "FALLOW DEER", "HORSE RIDER", "HUMAN", "JACKDAW", "LONGHORN CATTLE",
    "MAGPIE", "NO ANIMAL", "PHEASANT", "RABBIT", "RED DEER", "RED FOX", "ROE DEER"
]

# Initialises the Grad-CAM++ from pytorch_grad_cam
grad_cam = GradCAMPlusPlus(model=vgg16, target_layers=[vgg16.features[-1]])

# Transforms for the images for preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Crops the bottom part of the image
def crop_bottom_part(image, crop_height):
    height, width = image.shape[:2]
    if crop_height >= height:
        raise ValueError("Crop height is greater than the image height.")
    cropped_image = image[:height - crop_height, :]  # Crop the bottom part
    return cropped_image

# Loads the custom VGG16 that was trained, dict to save, dict to store predictions and pixels to crop 
class VGG16Classification:
    def __init__(self, model, processed_video_dir, detection_results, crop_height=0):
        self.model = model
        self.processed_video_dir = processed_video_dir
        self.detection_results = detection_results
        self.crop_height = crop_height  # Number of pixels to crop from the bottom

    # Classifies a image using the VGG16 
    def classify_image(self, img):
        img_transformed = data_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(img_transformed)
            probs = torch.softmax(output, dim=1)  # Get probabilities
            _, preds = torch.max(output, 1)
            return preds.item(), probs.squeeze().cpu().numpy(), img_transformed

    # Heatmap for the image
    def generate_gradcam(self, img_transformed, class_id, original_image):
        original_image_resized = original_image.resize((512, 512))
        target = [ClassifierOutputTarget(class_id)]
        grayscale_cam = grad_cam(input_tensor=img_transformed, targets=target)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(np.array(original_image_resized) / 255., grayscale_cam, use_rgb=True)
        return cam_image

    # Processes a video file where extracting frames at specific timestamps and classifying them   
    def process_video(self, video_path, folder, file_name):
        print(f"Processing video: {video_path}")
        self.detection_results[f"{folder}/{file_name}"] = defaultdict(int)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error - Could not open video {video_path}.")
            self.detection_results[f"{folder}/{file_name}"]['Error'] = 'Error - Could not open video'
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        target_times = [0, 10, 20, 30]  # Seconds
        for t in target_times:
            frame_index = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            # Crop the bottom part of the image if crop_height > 0
            if self.crop_height > 0:
                try:
                    frame = crop_bottom_part(frame, self.crop_height)
                except ValueError as ve:
                    print(f"Warning: {ve}. Skipping cropping.")
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            class_id, probs, img_transformed = self.classify_image(img)
            sorted_probs_idx = np.argsort(-probs)

            for idx in sorted_probs_idx:
                label = class_names[idx]
                prob = probs[idx]

                if prob > 0.1:  # Consider only classes with a probability higher than 10%
                    # Saves the Grad-CAM++ heatmaps 
                    cam_image = self.generate_gradcam(img_transformed, idx, img)
                    cam_image_path = os.path.join(self.processed_video_dir, f'{folder}-{file_name}-sec{t}-{label}.jpg')
                    cv2.imwrite(cam_image_path, cam_image)

                    # Saves the predicitons results
                    self.detection_results[f"{folder}/{file_name}"][(t, label)] += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished processing {video_path}")

# Run the VGG16 
def run_model(base_dir, processed_video_dir, crop_height=100):
    detection_results = defaultdict(lambda: defaultdict(int))
    output_csv = os.path.join(processed_video_dir, 'VGG16_Results.csv')
    process_videos(base_dir, processed_video_dir, detection_results, output_csv, crop_height)

# Processes all videos
def process_videos(base_dir, processed_video_dir, detection_results, output_csv, crop_height):
    print(f"Looking in base directory: {base_dir}")
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith('.mp4'):
                    video_path = os.path.join(folder_path, file)
                    classifier = VGG16Classification(vgg16, processed_video_dir, detection_results, crop_height)
                    classifier.process_video(video_path, folder, file)

    # Write classification results as CSV
    print(f"Saving predictions results to {output_csv}")
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['Video', 'FPS', 'Subject', 'Count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for video, detections in detection_results.items():
            for (time_sec, subject), count in detections.items():
                writer.writerow({
                    'Video': video,
                    'FPS': time_sec,
                    'Subject': subject,
                    'Count': count
                })

    print("Processing complete. CSV results saved to", output_csv)
