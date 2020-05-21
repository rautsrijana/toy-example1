import random
import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

eval_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])


class BallotDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, data_desc, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.df = pd.read_csv(data_desc)
        self.data_dir = data_dir
        self.filenames, self.labels = self.fetch_file_and_label()
        self.transform = transform

    def fetch_file_and_label(self):
        files = os.listdir(self.data_dir)
        filenames = [os.path.join(self.data_dir, f)
                for f in files if f.endswith('.jpeg')]
        names = [x.split("/")[-1] for x in filenames]
        labels = self.df[self.df['Data'].isin(names)]['Label'].tolist()
        return filenames, labels

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx]).convert("RGB")  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(
                        BallotDataset(path, 'data/train.csv', train_transformer), 
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=params.num_workers,
                        pin_memory=params.cuda)
            elif split == 'val':
                dl = DataLoader(
                        BallotDataset(path, 'data/train.csv', eval_transformer), 
                        batch_size=params.batch_size,
                        shuffle=False,
                        num_workers=params.num_workers,
                        pin_memory=params.cuda)
            else:
                dl = DataLoader(
                        BallotDataset(path, 'data/test.csv', eval_transformer), 
                        batch_size=params.batch_size,
                        shuffle=False,
                        num_workers=params.num_workers,
                        pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders


if __name__ == '__main__':
    dl = DataLoader(
            BallotDataset('data/256x256_testset/train_signs', 'data/train.csv', train_transformer),
            batch_size=4)

    for i, x in enumerate(dl):
        print(x)
        if i == 1:
            break

    
