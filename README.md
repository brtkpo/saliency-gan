# for CPU
pip install -r requirements.txt   
# for NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps

#Data
https://salicon.net/challenge-2017/
Images, Fixations and Fixation Maps

## Dataset
Download from [SALICON 2017](https://salicon.net/challenge-2017/) and place the extracted files in the `data/` folder.

Dataset structure:

```text
Saliency_GAN/
└── data/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── maps/
    │   ├── train/
    │   └── val/
    └── fixations/
        ├── train/
        ├── val/
        └── test/
```