import torch
import matplotlib.pyplot as plt
from dataset import SaliconDataset
from model import Generator

DATA_DIR = "data"
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_15_valL1_0.0713.pth"
# CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_20_valL1_0.0665.pth"
CHECKPOINT_PATH = "checkpoints/best_model.pth"

val_dataset = SaliconDataset(
    split="val",
    data_dir=DATA_DIR,
    img_size=IMG_SIZE
)

gen = Generator(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
gen.load_state_dict(checkpoint["gen_state_dict"])
gen.eval()

idx = 1112
img_name, image, sal_gt = val_dataset[idx]

image = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    sal_pred = gen(image)

image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
sal_gt_np = sal_gt.squeeze(0).cpu().numpy()
sal_pred_np = sal_pred.squeeze(0).squeeze(0).cpu().numpy()

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(image_np)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(sal_gt_np, cmap="hot")
plt.title("GT Saliency")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(sal_pred_np, cmap="hot")
plt.title("Predicted Saliency")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(image_np)
plt.imshow(sal_pred_np, cmap="hot", alpha=0.5)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()