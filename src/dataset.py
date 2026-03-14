import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import matplotlib

matplotlib.use("TkAgg")


class SaliconDataset(Dataset):
    """
    This class loads RGB images and their corresponding saliency maps,
    optionally applying simple data augmentations during training.
    """

    def __init__(
        self, split: str, data_dir: str, img_size: tuple[int, int], augment: bool
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        split : str,
            Dataset split ("train" or "val").
        data_dir : str,
            Root directory of the dataset.
        img_size : Tuple[int, int],
            Size to which images and saliency maps are resized.
        augment : bool,
            Whether to apply data augmentation (only used for training).
        """
        assert split in ["train", "val"], "split must be train or val for maps"
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.img_dir = os.path.join(data_dir, "images", split)
        self.map_dir = os.path.join(data_dir, "maps", split)

        self.images = sorted(os.listdir(self.img_dir))
        self.maps = sorted(os.listdir(self.map_dir))
        assert len(self.images) == len(self.maps), (
            f"Number of images and maps mismatch in split {split}"
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of image-saliency map pairs.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single dataset sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Tuple[str, torch.Tensor, torch.Tensor]
            A tuple containing:
            - image filename
            - image tensor of shape (3, H, W)
            - saliency map tensor of shape (1, H, W)
        """
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

        image_tensor = (
            torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
        )
        sal_tensor = torch.from_numpy(np.array(sal_map)[None, ...]).float() / 255.0

        return img_name, image_tensor, sal_tensor
