# Animal Classification and Action Recognition

## Installation

Install these required libraries and packages to run the code.

```bash
# OpenCV, Ultralytics, and PyTorch
python -m pip install opencv-python ultralytics torch torchvision torchaudio

# Install these versions of PyTorch, TorchVision, and TorchAudio
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Install DeepSORT for ID and tracking
pip install deep_sort_realtime

# PyTorchVideo 
pip install pytorchvideo

# Additional libraries
pip install scikit-learn
pip install openpyxl
pip install grad-cam
pip install --upgrade tensorflow-hub
```

There can be an error within the pytorchvideo\transforms\augmentations.py file where:

You have to change this
```python
import torchvision.transforms.functional_tensor as F_t
```
To this:
```python
import torchvision.transforms.functional as F_t
```


