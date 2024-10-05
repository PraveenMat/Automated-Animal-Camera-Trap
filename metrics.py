import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Plots confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    labels = sorted(set(y_true).union(set(y_pred)))  # Union of true and predicted labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Loads CSV 
vgg16_results = pd.read_csv('Predictions/VGG16_Results.csv')
yolo_results = pd.read_csv('Predictions/YOLOv8_Results.csv')
test_data = pd.read_csv('Predictions/Test Data V2.csv')


# Standardise labels in all datasets to uppercase for consistency
vgg16_results['Subject'] = vgg16_results['Subject'].str.upper()
yolo_results['Subject'] = yolo_results['Subject'].str.upper()
test_data['Subject'] = test_data['Subject'].str.upper()

# VGG16 mapping the labels
vgg_label_mapping = {
    'HUMAN': 'PERSON',
    'NO ANIMAL': 'NA',
}

# Applys the label mapping to VGG16 results
vgg16_results['Mapped Subject'] = vgg16_results['Subject'].map(vgg_label_mapping).fillna(vgg16_results['Subject'])

# Group by Video and Mapped Subject and count
vgg16_video_level = vgg16_results.groupby(['Video', 'Mapped Subject']).size().reset_index(name='Counts')

# Merges with the ground truth data
vgg16_merged = pd.merge(vgg16_video_level, test_data, how='left', left_on=['Video', 'Mapped Subject'], right_on=['Video', 'Subject'])

# Ensure labels are consistent and handle any NaNs
vgg16_merged['Mapped Subject'] = vgg16_merged['Mapped Subject'].fillna('NA').astype(str)
vgg16_merged['Subject'] = vgg16_merged['Subject'].fillna('NA').astype(str)

# Check if the predicted subject matches with the true subject
vgg16_merged['Subject_Correct'] = vgg16_merged['Mapped Subject'] == vgg16_merged['Subject']

# Replace NaNs in count columns with 0
vgg16_merged['Counts'] = vgg16_merged['Counts'].fillna(0).astype(int)
vgg16_merged['Subject Count'] = vgg16_merged['Subject Count'].fillna(0).astype(int)

# Calculates the VGG16 Count Accuracy
vgg16_merged['Count_Correct'] = vgg16_merged['Counts'] == vgg16_merged['Subject Count']
vgg_count_accuracy = vgg16_merged['Count_Correct'].mean()

# Calculates the VGG Count Precision, Recall, F1
vgg_count_precision, vgg_count_recall, vgg_count_f1, _ = precision_recall_fscore_support(
    vgg16_merged['Subject Count'], vgg16_merged['Counts'], average='macro', zero_division=1)

# True labels and predicted labels for VGG16
vgg_true_labels = vgg16_merged['Subject']
vgg_pred_labels = vgg16_merged['Mapped Subject']

# Calculates accuracy, precision, recall, f1 score for VGG16 subject 
vgg_precision, vgg_recall, vgg_f1, _ = precision_recall_fscore_support(vgg_true_labels, vgg_pred_labels, average='macro', zero_division=1)
vgg_accuracy = accuracy_score(vgg_true_labels, vgg_pred_labels)

# Plot confusion matrix for VGG16 Animal subject
plot_confusion_matrix(vgg_true_labels, vgg_pred_labels, "VGG16 Animal Classifcation Confusion Matrix")

# YOLOv8 mapping the labels
label_mapping = {
    'COW': 'LONGHORN CATTLE',
}

# Applys the label mapping to the YOLO results
yolo_results['Mapped Subject'] = yolo_results['Subject'].map(label_mapping).fillna(yolo_results['Subject'])

# Convert YOLO predicted labels to uppercase to match the true labels
yolo_results['Mapped Subject'] = yolo_results['Mapped Subject'].str.upper()

# Ensure the true labels in the test data are also uppercase
test_data['Subject'] = test_data['Subject'].fillna('NA').astype(str)

# Group by Video, Mapped Subject, and Animal ID
yolo_last_counts = yolo_results.groupby(['Video', 'Mapped Subject', 'Animal ID']).last().reset_index()
yolo_final_counts = yolo_last_counts.groupby(['Video', 'Mapped Subject'])['Count'].max().reset_index()

# Merges YOLOv8 results with ground truth data
yolo_merged_final = pd.merge(yolo_final_counts, test_data, how='left', left_on=['Video', 'Mapped Subject'], right_on=['Video', 'Subject'])

# Checks if the predicted subject matches with the true subject
yolo_merged_final['Subject_Correct'] = yolo_merged_final['Mapped Subject'] == yolo_merged_final['Subject']

# Replaces the NaNs in count columns with 0
yolo_merged_final['Subject Count'] = yolo_merged_final['Subject Count'].fillna(0).astype(int)
yolo_merged_final['Count'] = yolo_merged_final['Count'].fillna(0).astype(int)

# True labels and predicted labels for YOLOv8
yolo_true_labels = yolo_merged_final['Subject']
yolo_pred_labels = yolo_merged_final['Mapped Subject']

# Ensure all labels are strings and handles NaN values
yolo_true_labels = yolo_true_labels.fillna('NA').astype(str)
yolo_pred_labels = yolo_pred_labels.fillna('NA').astype(str)

# Calculate precision, recall, f1 score, and accuracy for YOLOv8 Subject
yolo_precision, yolo_recall, yolo_f1, _ = precision_recall_fscore_support(yolo_true_labels, yolo_pred_labels, average='macro', zero_division=1)
yolo_accuracy = accuracy_score(yolo_true_labels, yolo_pred_labels)

# Plots confusion matrix for YOLOv8 Animal Subject
plot_confusion_matrix(yolo_true_labels, yolo_pred_labels, "YOLOv8 Animal Classifcation Confusion Matrix")

# Calculates count accuracy for YOLOv8
yolo_merged_final['Count_Correct'] = yolo_merged_final['Count'] == yolo_merged_final['Subject Count']
yolo_count_accuracy_final = yolo_merged_final['Count_Correct'].mean()

# Calculates YOLO Count Precision, Recall, F1
yolo_count_precision, yolo_count_recall, yolo_count_f1, _ = precision_recall_fscore_support(
    yolo_merged_final['Subject Count'], yolo_merged_final['Count'], average='macro', zero_division=1)

# Plot confusion matrix for VGG16 Count
plot_confusion_matrix(vgg16_merged['Subject Count'], vgg16_merged['Counts'], "VGG16 Count Confusion Matrix")

# Plots confusion matrix for YOLOv8 Count
plot_confusion_matrix(yolo_merged_final['Subject Count'], yolo_merged_final['Count'], "YOLOv8 Count Confusion Matrix")

# Prepares data for table
metrics_table = [
    ['VGG16', vgg_accuracy, vgg_precision, vgg_recall, vgg_f1, vgg_count_accuracy, vgg_count_precision, vgg_count_recall, vgg_count_f1],
    ['YOLOv8', yolo_accuracy, yolo_precision, yolo_recall, yolo_f1, yolo_count_accuracy_final, yolo_count_precision, yolo_count_recall, yolo_count_f1]
]

# Define headers
headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Count Accuracy", "Count Precision", "Count Recall", "Count F1 Score"]

# Print the table
print(tabulate(metrics_table, headers=headers, tablefmt="grid"))
