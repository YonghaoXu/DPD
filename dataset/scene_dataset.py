from torch.utils.data import Dataset
from PIL import Image
import os

def default_loader(path):
    return Image.open(path).convert('RGB')

class scene_dataset(Dataset):
    def __init__(self, root_dir, pathfile=None, lines=None, transform=None, loader=default_loader):
        """
        Args:
            root_dir (str): Root path to data
            pathfile (str): Path to txt file with "image_path label"
            lines (list): Optional list of "image_path label" strings
            transform: torchvision transform
            loader: image loading function
        """
        self.imgs = []
        self.transform = transform
        self.loader = loader
        self.root_dir = root_dir

        if pathfile is not None:
            with open(pathfile, 'r') as pf:
                lines = pf.readlines()

        if lines is None:
            raise ValueError("Either pathfile or lines must be provided")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            label = int(parts[1])
            name = os.path.splitext(os.path.basename(img_path))[0]

            if os.path.isabs(img_path):
                full_path = img_path
            else:
                full_path = os.path.join(root_dir, img_path)

            self.imgs.append((full_path, label, name))

    def __getitem__(self, index):
        path, label, name = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, name

    def __len__(self):
        return len(self.imgs)
