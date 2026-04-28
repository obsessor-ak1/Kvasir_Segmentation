from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.tv_tensors import Image, Mask


class Kvasir1Dataset(Dataset):
    """Implementation of the Kvasir1 Dataset for medical image segmentation."""
    def __init__(self, root_path:str, transform=None, mode='train'):
        self.root_path = Path(root_path)
        self.transform = transform
        self.mode = mode
        image_set = self.root_path / f"{"train.txt" if mode=='train' else "val.txt"}"
        with open(image_set, 'r') as f:
            self.image_names = f.readlines()
        self.image_names = [img.strip() for img in self.image_names]
        self.image_dir = self.root_path / "Kvasir-SEG/Kvasir-SEG/images"
        self.mask_dir = self.root_path / "Kvasir-SEG/Kvasir-SEG/masks"

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.image_dir / f"{image_name}.jpg"
        mask_path = self.mask_dir / f"{image_name}.jpg"
        image = Image(decode_image(image_path))
        mask = Mask(decode_image(mask_path))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask
