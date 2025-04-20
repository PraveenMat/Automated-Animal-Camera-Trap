# Automated Camera‑Trap Video Processing with Deep Learning

> **Automated Approach to Processing Camera‑Trap Data Using Machine Learning**  

This project automates the _detection_, _counting_ and _behaviour analysis_ of wildlife captured by camera‑traps. It bundles two complementary deep‑learning pipelines, a simple Tkinter GUI, utilities for data preparation and several analysis notebooks/scripts—enabling ecologists to process thousands of hours of footage in a fraction of the time normally required.

---
## Key Features
| Module | Tech | Purpose |
|--------|------|---------|
| **Approach 1 – VGG16** | PyTorch | Frame‑level animal **classification** + **counting**; Grad‑CAM++ heat‑maps for explainability |
| **Approach 2 – YOLOv8 x + DeepSORT + SlowFast** | Ultralytics YOLO · deep_sort_realtime · PyTorchVideo | Real‑time **detection**, multi‑object **tracking**, approximate **counting** & **behaviour recognition** |
| **GUI** | Tkinter | Folder‑based batch processing with model selector |
| **Metrics** | scikit‑learn, seaborn | Confusion matrices + accuracy / precision / recall / F1 reports |
| **Utilities** | OpenCV, pandas | Video‑to‑frame extractor, dataset split helper, training script |


---
## Example
![YOLO processed video](GIF/ex1.gif)
---
## Getting Started
### Install dependencies
> Tested with Python 3.10 and CUDA 11.8
```bash
pip install opencv-python ultralytics torch torchvision torchaudio
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install deep_sort_realtime pytorchvideo scikit-learn openpyxl grad-cam
```
> **Heads‑up:**  A minor bug in `pytorchvideo` requires one line to be patched:
> ```python
> # .../site-packages/pytorchvideo/transforms/augmentations.py
> # change
> import torchvision.transforms.functional_tensor as F_t
> # to
> import torchvision.transforms.functional as F_t
> ```

### Results (51 test videos)

YOLOv8x outperforms VGG16 overall, especially in detecting multiple animals per frame, but still struggles with small, distant or night‑time targets. Behaviour recognition is limited by the use of human‑action labels.

---
## Limitations & Future Work
* **Bounding‑box ground‑truth** is required to fine‑tune YOLO and further boost precision.
* Replace human Kinetics‑400 labels with an animal‑specific action dataset for SlowFast.
* Migrate to **YOLOv9** or **RT‑DETR** for small‑object performance.
* Optimise the GUI for batch/job‑queue processing and add a progress bar.

---
## License
This repository is released under the **MIT License** –