"""PyTorch Dataset, transforms and class mapping for LeafDoc."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

IMG_SIZE = 128  # CPU par tez training ke liye chhota size
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def get_classes(split_dir="data/splits"):
    df = pd.read_csv(f"{split_dir}/train.csv")
    return sorted(df.label.unique())


class LeafDataset(Dataset):
    def __init__(self, csv_path, classes, transform):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), self.class_to_idx[row.label]


def get_loaders(batch_size=32, split_dir="data/splits"):
    classes = get_classes(split_dir)
    sets = {
        "train": LeafDataset(f"{split_dir}/train.csv", classes, train_tf),
        "val": LeafDataset(f"{split_dir}/val.csv", classes, eval_tf),
        "test": LeafDataset(f"{split_dir}/test.csv", classes, eval_tf),
    }
    loaders = {
        k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"), num_workers=0)
        for k, v in sets.items()
    }
    return loaders, classes


if __name__ == "__main__":
    loaders, classes = get_loaders()
    x, y = next(iter(loaders["train"]))
    print(len(classes), "classes")
    print(x.shape, y[:8])