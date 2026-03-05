pip install uv
uv sync

uv run main.py -c config.json

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