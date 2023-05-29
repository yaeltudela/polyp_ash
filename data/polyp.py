import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

classes_ass = {
    'AD': 0,
    'HP': 1,
    'ASS': 2
}

classes_bin = {
    'AD': 0,
    'HP': 1,
    'ASS': 1
}


class PolypCls(Dataset):
    def __init__(self, root_dir, csv_file, split='train', use_ass=False, only_polyp=False, transforms=None,
                 load_in_memory=True):
        self.dataset_name = root_dir.split("/")[-1]
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.only_polyp = only_polyp

        self.challenge = True
        if use_ass:
            self.challenge = False

        self.num_classes = 3 if use_ass and not self.challenge else 2

        assert not (use_ass and self.challenge), "Challenge is only AD, NAD"

        # Load GT
        df = pd.read_csv(f'{root_dir}/m_{split}/{csv_file}')

        self.df = df

        self.image_id = df.image_id.astype('str')
        self.labels = df.cls.astype('str')
        self.labels = self.labels.apply(lambda x: classes_ass[x] if use_ass else classes_bin[x])

        if self.challenge and self.split != 'test':
            self.labels = (~df.Histologia.astype('bool')).astype('int')

        # increase labels by 1 to make 0 the background class
        self.labels += 1

        self.load_in_memory = load_in_memory
        if load_in_memory:
            stored_annots, stored_imgs, stored_labels = self.load_all_data()
            self.stored_imgs = stored_imgs
            self.stored_annots = stored_annots
            self.stored_labels = stored_labels

    def load_all_data(self):
        stored_imgs = []
        stored_annots = []
        stored_labels = []
        for i in range(self.image_id.shape[0]):
            img = self.load_image(i)
            annot, label = self.load_annots(i)

            stored_imgs.append(img)
            stored_annots.append(annot)
            stored_labels.append(label)
        return stored_annots, stored_imgs, stored_labels

    def __len__(self):
        return self.image_id.shape[0]

    def __getitem__(self, index):
        if self.load_in_memory:
            image = self.stored_imgs[index]
            annot = self.stored_annots[index]
            label = self.stored_labels[index]
        else:
            image = self.load_image(index)
            annot, label = self.load_annots(index)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=annot)
            image, annot = transformed['image'], transformed['mask']
        else:
            image = np.array(image)

        return image, label, annot

    def load_annots(self, index):
        annot_path = f"{self.root_dir}/m_{self.split}/masks/{index}.tif"
        label = self.labels[index]
        if self.only_polyp:
            label = 1
        annot_prime = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) // 255
        annot = annot_prime * label

        return annot, label

    def load_image(self, index, to_rgb=True):
        data_path = f"{self.root_dir}/m_{self.split}/images/{index}.tif"
        image = cv2.imread(data_path)
        if to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def compute_sample_weights(self):
        weights = self.labels.value_counts().max() / self.labels.value_counts()

        # weights = 1 - (self.labels.value_counts() / self.labels.shape[0])
        return torch.tensor(weights.values)[self.labels - 1]

    def compute_class_weights(self, add_bg=False):
        normalize = True
        weights = self.labels.value_counts().values
        weights = 1 / np.log(weights)
        if add_bg:
            out_weights = torch.zeros(weights.shape[0] + 1)
            out_weights[0] = weights.min()
            out_weights[1:] = torch.tensor(weights)
        else:
            out_weights = torch.Tensor(weights)

        if normalize:
            out_weights = out_weights / out_weights.sum()

        print(out_weights)

        return out_weights

    def __str__(self):
        return f"Polyp Dataset\n{len(self)} samples\n{self.labels.unique()}"


class PolypSegm(Dataset):
    def __init__(self, root_dir, transforms=None, load_in_memory=True, im_extension='bmp', annot_extension='tif'):
        self.dataset_name = root_dir.split("/")[-1]
        self.root_dir = root_dir
        self.transforms = transforms

        self.split = 'test'

        self.num_classes = 1

        # Load GT
        self.image_files = glob.glob(f"{self.root_dir}/images/*.{im_extension}")
        self.annot_files = [i.replace(f'.{im_extension}', f'.{annot_extension}').replace('images', 'masks') for i in
                            self.image_files]

        self.load_in_memory = load_in_memory
        if load_in_memory:
            self.stored_annots, self.stored_imgs, self.stored_labels = self.load_all_data()

    def load_all_data(self):
        stored_imgs = []
        stored_annots = []
        stored_labels = []
        for image_file, annot_file in zip(self.image_files, self.annot_files):
            img = self.load_image(image_file)
            annot, label = self.load_annots(annot_file)

            stored_imgs.append(img)
            stored_annots.append(annot)
            stored_labels.append(label)
        return stored_annots, stored_imgs, stored_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        if self.load_in_memory:
            image = self.stored_imgs[index]
            annot = self.stored_annots[index]
            label = self.stored_labels[index]
        else:
            image = self.load_image(index)
            annot, label = self.load_annots(index)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=annot)
            image, annot = transformed['image'], transformed['mask']
        else:
            image = np.array(image)

        return image, label, annot

    def load_annots(self, annot_path):
        label = 1
        annot_prime = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) // 255
        annot = annot_prime * label

        return annot, label

    def load_image(self, data_path):
        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class KvasirSegmentationDataset(Dataset):
    def __init__(self, path, split="train", transforms=None):
        super().__init__()
        self.path = os.path.join(path, "segmented-images/")
        self.fnames = os.listdir(os.path.join(self.path, "images"))
        self.dataset_name = 'KVASIR SEG'

        # deterministic partition
        self.split = split
        train_size = int(len(self.fnames) * 0.7)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size

        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]

        self.transforms = transforms

        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "valid":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        data_path = f"{self.path}images/{self.split_fnames[index]}"
        annot_path = f"{self.path}masks/{self.split_fnames[index]}"

        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1
        annot_prime = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE) // 255
        annot = annot_prime * label

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=annot)
            image, annot = transformed['image'], transformed['mask']

        return image, label, annot
