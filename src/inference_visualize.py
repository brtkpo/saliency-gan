import torch
import matplotlib.pyplot as plt

from .dataset import SaliconDataset
from .model import Generator

def run_inference_visualize(
    data_dir,
    checkpoint_path,
    img_size=(224, 224),
    split="val",
    idx=5,
    device=None,
):

    gen = Generator(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    gen.eval()

    dataset = SaliconDataset(
        split=split,
        data_dir=data_dir,
        img_size=img_size
    )

    img_name, image, gt_map = dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)

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
    plt.show()

#save_path = "saliency_result_img5.png"
#plt.savefig(save_path, bbox_inches='tight', dpi=150)

# if __name__ == "__main__":
#     run_inference_visualize()