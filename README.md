# UPM158 - AI for detecting fake/modified images

## Installation

Python 3.11 should be installed.

```bash
python3 -m venv .venv
# Bash
source .venv/bin/activate
# Windows
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Cuda
Problems can occur with Pytorch installation. Please visit the website https://pytorch.org/ to install the correct version for your system.  
If you do not CUDA compatible GPU, you can install the CPU version of Pytorch, and change the device in the code to CPU.
