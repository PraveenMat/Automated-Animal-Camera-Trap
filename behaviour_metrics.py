import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Loads CSV
yolo_results = pd.read_csv('Predictions/YOLOv8_Results.csv')
test_data = pd.read_csv('Predictions/Test Data V2.csv')

# Plots confusion matrix for YOLOv8 behaviour using the true labels
def plot_confusion_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Standardise labels in all datasets to uppercase for consistency
yolo_results['Subject'] = yolo_results['Subject'].str.upper()
yolo_results['Behaviour'] = yolo_results['Behaviour'].str.upper().fillna('NA')  # Fills NaNs with 'NA'
test_data['Subject'] = test_data['Subject'].str.upper()
test_data['Behaviour'] = test_data['Behaviour'].str.upper().fillna('NA')  # Fills NaNs with 'NA'

# Merges YOLOv8 results with ground truth data on Video & Subject
yolo_behaviour_merged = pd.merge(yolo_results, test_data, how='left', left_on=['Video', 'Subject'], right_on=['Video', 'Subject'])

# True labels and predicted labels for behaviour
yolo_true_behaviours = yolo_behaviour_merged['Behaviour_y'].fillna('NA').astype(str)
yolo_pred_behaviours = yolo_behaviour_merged['Behaviour_x'].fillna('NA').astype(str)

# Define the full set of true labels
true_labels_updated = ['WALKING', 'STANDING', 'GRAZING', 'RUNNING', 'SCRATCHING', 'LAYING DOWN', 'FORAGING', 'NA']

# Calculates the precision, recall, f1 score, and accuracy including with 'NA' for YOLOv8 behaviour predictions
precision_original, recall_original, f1_original, _ = precision_recall_fscore_support(
    yolo_true_behaviours, yolo_pred_behaviours, labels=true_labels_updated, average='macro', zero_division=1)
accuracy_original = accuracy_score(yolo_true_behaviours, yolo_pred_behaviours)

# Prepare the data to display on the tabel
metrics_data_original = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy_original, precision_original, recall_original, f1_original]
}

# Prints for the true labels only metrics
metrics_df_original = pd.DataFrame(metrics_data_original)
metric = tabulate(metrics_df_original, headers='keys', tablefmt='grid', showindex=False)
print(metric)

# Plots confusion matrix using only the true labels
plot_confusion_matrix(yolo_true_behaviours, yolo_pred_behaviours, 'YOLOv8 Behaviour Identification Confusion Matrix', true_labels_updated)
