import torch
import matplotlib.pyplot as plt

from dataset import SaliconDataset
from model import Generator

DATA_DIR = "data"
IMG_SIZE = (224, 224)
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_15_valL1_0.0713.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
gen.load_state_dict(checkpoint["gen_state_dict"])
gen.eval()

test_dataset = SaliconDataset(
    split="val",
    data_dir=DATA_DIR,
    img_size=IMG_SIZE
)

img_name, image, gt_map = test_dataset[5]
image_tensor = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    saliency = gen(image_tensor)

saliency = saliency.squeeze().cpu().numpy()
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


gt_map = gt_map.squeeze().cpu().numpy()
gt_map = (gt_map - gt_map.min()) / (gt_map.max() - gt_map.min() + 1e-8)


image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
plt.title("Original image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Ground Truth (GT)")
plt.imshow(gt_map, cmap="hot")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Predicted saliency")
plt.imshow(saliency, cmap="hot")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Overlay (Pred)")
plt.imshow(image_np)
plt.imshow(saliency, cmap="hot", alpha=0.5)
plt.axis("off")

plt.tight_layout()

#save_path = "saliency_result_img5.png"
#plt.savefig(save_path, bbox_inches='tight', dpi=150)