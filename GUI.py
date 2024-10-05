import tkinter as tk
from tkinter import filedialog
import Model1 # VGG16 model 
import Model2  # YOLOv8 model 
import ctypes
import sys

# To get rid of the blurryness when the GUI is open due to DPI
if sys.platform.startswith('win'):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        print({str(e)})

# Select the folder
def select_folder():
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)

# Folder to save the processed video and CSV
def select_output_folder():
    output_folder_selected = filedialog.askdirectory()
    output_folder_path.set(output_folder_selected)

# Run the GUI
def run_model():
    base_dir = folder_path.get()
    processed_video_dir = output_folder_path.get()
    selected_model = model_var.get()

    # Checks if both the input and output folders are selected
    if base_dir and processed_video_dir:
        try:
            if selected_model == "VGG16":
                Model1.run_model(base_dir, processed_video_dir)
            elif selected_model == "YOLOv8":
                Model2.run_model(base_dir, processed_video_dir)
            else:
                error_label.config(text="Selected model not recognised", fg="red") 
        except Exception as e:
            error_label.config(text=f"Error: {str(e)}", fg="red")
    else:
        error_label.config(text="Please select both a folder to process and an output folder.", fg="red")

# Hover effects
def on_enter(event):
    event.widget['background'] = '#2980b9'

def on_leave(event):
    event.widget['background'] = '#3498db'

# Tkinter GUI
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("800x800")
root.configure(bg="#003B49")  # main background colour

# Stores the folder paths and selected model
folder_path = tk.StringVar()
output_folder_path = tk.StringVar()
model_var = tk.StringVar(value="YOLOv8")  # Default model is YOLOv8 due to it high performance

# Layout of the interface 
label = tk.Label(root, text="Select Folder to Process:", bg="#003B49", fg="white", font=("Arial", 12))
label.pack(pady=10)

# Select folder button
select_button = tk.Button(root, text="Select Folder", command=select_folder, bg="#3498db", fg="white", font=("Arial", 10), relief="flat")
select_button.bind("<Enter>", on_enter)
select_button.bind("<Leave>", on_leave)
select_button.pack(pady=10)

folder_display = tk.Entry(root, textvariable=folder_path, width=50, bg="#ecf0f1", relief="flat")
folder_display.pack(pady=10)

# Folder output
label_output = tk.Label(root, text="Select Folder to Save Processed Videos:", bg="#003B49", fg="white", font=("Arial", 12))
label_output.pack(pady=10)

select_output_button = tk.Button(root, text="Select Output Folder", command=select_output_folder, bg="#3498db", fg="white", font=("Arial", 10), relief="flat")
select_output_button.bind("<Enter>", on_enter)
select_output_button.bind("<Leave>", on_leave)
select_output_button.pack(pady=10)

output_folder_display = tk.Entry(root, textvariable=output_folder_path, width=50, bg="#ecf0f1", relief="flat")
output_folder_display.pack(pady=10)

# Model select
label_model = tk.Label(root, text="Select Model:", bg="#003B49", fg="white", font=("Arial", 12))
label_model.pack(pady=10)

# Dropdown menu for model selecting a model
model_dropdown = tk.OptionMenu(root, model_var, "YOLOv8", "VGG16") 
model_dropdown.config(bg="#3498db", fg="white", font=("Arial", 10), relief="flat")
model_dropdown["menu"].config(bg="#3498db", fg="white")
model_dropdown.pack(pady=10)

# Run button
run_button = tk.Button(root, text="Run Model", command=run_model, bg="#27ae60", fg="white", font=("Arial", 12, "bold"), relief="flat")
run_button.pack(pady=20)

error_label = tk.Label(root, text="", bg="#003B49", font=("Arial", 10))
error_label.pack()

root.mainloop()