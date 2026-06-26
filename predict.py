import argparse
import random

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

from seg_modules.data import Kvasir1Dataset
from seg_modules.architectures.unet import UNet, AttentionUNet, UNetPlusPlus

IMAGE_SIZE = (256, 256)

model_catalog = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unet_plus_plus": UNetPlusPlus,
}

val_transform = v2.Compose(
    [v2.Resize(IMAGE_SIZE, antialias=True), v2.ToDtype(torch.float32, scale=True)]
)


def load_model(
    model_name: str, checkpoint_path: str, device: torch.device, depth: int = 5
) -> torch.nn.Module:
    """Loads a segmentation model by architecture name and checkpoint path."""
    if model_name not in model_catalog:
        raise ValueError(
            f"Model '{model_name}' not recognized. Choose from: {list(model_catalog.keys())}"
        )

    model = model_catalog[model_name](in_channels=3, num_classes=1, depth=depth)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(
            f"Expected key 'model' in checkpoint, but found keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def get_random_samples(dataset_path: str, count: int, seed: int = None):
    """Loads the validation dataset, samples random images/masks, and returns batch tensors."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Loading validation dataset from: {dataset_path}")
    val_dataset = Kvasir1Dataset(
        root_path=dataset_path, transform=val_transform, mode="val"
    )

    dataset_size = len(val_dataset)
    if count <= 0:
        raise ValueError("Count must be greater than 0.")
    if count > dataset_size:
        print(
            f"Warning: requested count ({count}) is larger than validation set size ({dataset_size}). Processing all {dataset_size} images."
        )
        count = dataset_size

    indices = random.sample(range(dataset_size), count)
    print(f"Selected random indices: {indices}")

    images = []
    masks = []
    for idx in indices:
        image, mask = val_dataset[idx]
        images.append(image)
        masks.append(mask)

    images_batch = torch.stack(images)
    masks_batch = torch.stack(masks)
    return images_batch, masks_batch, indices


def predict(
    model, images: torch.Tensor, device, threshold: float = 0.5
) -> torch.Tensor:
    """Predicts masks for a batch of images using the specified model and threshold."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    images = images.to(device)

    with torch.inference_mode():
        pred_maps = model(images)
    pred_maps = torch.sigmoid(pred_maps)
    pred_masks = pred_maps >= threshold

    return pred_masks.squeeze(dim=1)


def plot_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    indices: list[int],
    output_path: str = "predictions.png",
):
    """Plots and saves rows of images, actual masks, and predicted masks."""
    print("Generating matplotlib plot...")
    count = images.shape[0]
    fig, axes = plt.subplots(count, 3, figsize=(12, 4 * count))

    # Ensure axes is a 2D array even if count is 1
    if count == 1:
        axes = axes.reshape(1, 3)

    for i in range(count):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Image {indices[i]}")
        axes[i, 0].axis("off")

        mask_np = masks[i].squeeze().cpu().numpy()
        actual_mask = mask_np > 0.5 if mask_np.max() <= 1.0 else mask_np > 127
        axes[i, 1].imshow(actual_mask, cmap="gray")
        axes[i, 1].set_title(f"Actual Mask {indices[i]}")
        axes[i, 1].axis("off")

        pred_mask_np = predictions[i].cpu().numpy()
        axes[i, 2].imshow(pred_mask_np, cmap="gray")
        axes[i, 2].set_title(f"Predicted Mask {indices[i]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved visualization plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation model on random validation images."
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=["unet", "attention_unet", "unet_plus_plus"],
        help="The name of the model architecture to load.",
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="The path to the model checkpoint file."
    )
    parser.add_argument(
        "dataset_path", type=str, help="The path to the dataset directory."
    )
    parser.add_argument(
        "count", type=int, help="The number of random images to process."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for random number generators to ensure reproducibility.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Depth of the network model (default: 5)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_name, args.checkpoint_path, device, depth=args.depth)

    images_batch, masks_batch, indices = get_random_samples(
        args.dataset_path, args.count, seed=args.seed
    )

    print("Running predictions on batch...")
    predictions = predict(model, images_batch, device)

    plot_predictions(images_batch, masks_batch, predictions, indices)


if __name__ == "__main__":
    main()
