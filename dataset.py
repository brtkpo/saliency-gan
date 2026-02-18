import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class SaliconDataset(Dataset):
    def __init__(self, split="train", data_dir="data", img_size=(224, 224), augment=False):
        assert split in ["train", "val"], "split must be train or val for maps"
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.img_dir = os.path.join(data_dir, "images", split)
        self.map_dir = os.path.join(data_dir, "maps", split)

        self.images = sorted(os.listdir(self.img_dir))
        self.maps = sorted(os.listdir(self.map_dir))
        assert len(self.images) == len(self.maps), f"Number of images and maps mismatch in split {split}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        map_path = os.path.join(self.map_dir, self.maps[idx])

        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        sal_map = Image.open(map_path).convert("L").resize(self.img_size)

        if self.augment and self.split == "train":
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                sal_map = sal_map.transpose(Image.FLIP_LEFT_RIGHT)

            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR)
            sal_map = sal_map.rotate(angle, resample=Image.BILINEAR)

        image_tensor = torch.from_numpy(np.array(image).transpose(2,0,1)).float() / 255.0
        sal_tensor = torch.from_numpy(np.array(sal_map)[None, ...]).float() / 255.0

        return img_name, image_tensor, sal_tensor

    def visualize(self, idx):
        img_name, image_tensor, sal_tensor = self[idx]
        image = np.transpose(image_tensor.numpy(), (1,2,0))
        sal_map = sal_tensor.squeeze(0).numpy()

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title(f"{img_name}")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(sal_map, cmap="hot")
        plt.title(f"Saliency map")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(image)
        plt.imshow(sal_map, cmap="hot", alpha=0.5)
        plt.title(f"{img_name} + overlay")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    dataset = SaliconDataset(split="train", data_dir="data", img_size=(224,224))
    for i in range(min(10, len(dataset))):
        dataset.visualize(i)
