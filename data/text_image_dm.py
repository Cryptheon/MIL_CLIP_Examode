# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, choice

import pandas as pd
import os
import pickle
import lmdb
import PIL
import argparse
import clip
import torch
import json
import numpy as np
#import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
import torch.nn as nn

class TextImageDataset(Dataset):
    def __init__(self,
                 lmdb_patches_path: str,
                 wsi_to_diagnosis: dict,
                 listed_data: list,
                 image_size=224,
                 set_size=32,
                 shuffle=True,
                 training_phase=True,
                 validation_labels=None
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__()
        self.lmdb_patches_path = lmdb_patches_path
        self.wsi_to_diagnosis = wsi_to_diagnosis
        self.listed_data = listed_data
        self.shuffle = shuffle
        self.image_size = image_size
        self.set_size = set_size
        self.training_phase = training_phase
        self.validation_labels = validation_labels

        print("Number of Images: ", len(self.listed_data))

    def __len__(self):
        return len(self.listed_data)


    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):

        try:
            key = self.listed_data[ind]
        except:
            return self.skip_sample(ind)

        # 10% of data misses its correspondent diagnosis
        # so in the case we miss a diagnosis and we are training, just proceed
        # to the next example, else if we are not training, use text="nothing"
        # as surrogate, because during evaluation we are more interested in
        # the average precision anyway
        if self.training_phase:
            try:
                text = self.wsi_to_diagnosis[key.split(".")[0]]
            except:
                return self.skip_sample(ind)
        else:
            text = ""

        # open lmdb environment for the embeddings
        try:
            env = lmdb.open(str(Path(self.lmdb_patches_path) / f"{Path(key)}"), readonly=True)
        except:
            return self.skip_sample(ind)

        entries = env.stat()["entries"]

        if entries == 0:
            return self.skip_sample(ind)

        # IMAGE EMBEDDINGS

        # this one is still random but we keep it as such for now
        # TODO: if shuffle false no random chosen patches.
        chosen_patches = np.random.randint(0,entries,self.set_size)
        img_latents = []
        # Start a new read transaction
        with env.begin() as txn:
            for chosen_patch in chosen_patches:
                # Read all images in one single transaction, with one lock
                data = txn.get(f"{chosen_patch:08}".encode("ascii"))
                image = pickle.loads(data)
                # Need to supply correct type, these are continuous latents
                img = np.fromstring(image, dtype=np.float32)
                img_latents.append(torch.tensor(img))

        env.close()

        # TEXTS
        description = text.split('\n')
        tokenized_text = clip.tokenize(description)[0]

        # during evaluation we want to know, where our latents come from
        if not self.training_phase:
            return img_latents, key, self.validation_labels[ind]

        return img_latents, tokenized_text

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 lmdb_patches_path: str,
                 wsi_to_diagnosis_path: str,
                 listed_data_path: str,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 shuffle=True,
                 set_size=32,
                 train_folds=[1,2]
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.lmdb_patches_path = lmdb_patches_path
        self.listed_data_path = listed_data_path
        self.wsi_to_diagnosis_path = wsi_to_diagnosis_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.shuffle = shuffle
        self.set_size = set_size
        self.train_folds = train_folds

        train_folds = [int(digit_fold) for digit_fold in train_folds]
        lmdb_patches = os.listdir(lmdb_patches_path)
        lmdb_patches = {lmdb_patch.split(".")[0]: lmdb_patch for lmdb_patch in lmdb_patches}

        with open(wsi_to_diagnosis_path, "r") as f:
            self.wsi_to_diagnosis = json.load(f)

        listed_data_df = pd.read_csv(listed_data_path)
        # choose which folds to use, testing fold is in a separate file
        training_folds = listed_data_df[listed_data_df["Fold"].isin(train_folds)]
        # the remaning folds are for validation
        validation_folds = list(set(range(0,10)) - set(train_folds))
        validation_folds = listed_data_df[listed_data_df["Fold"].isin(validation_folds)]

        self.listed_train_data = training_folds.WSI.tolist()
        self.listed_train_data = [lmdb_patches[wsi.split(".")[0]] for wsi in self.listed_train_data if wsi.split(".")[0] in lmdb_patches]

        self.listed_val_data = validation_folds.WSI.tolist()
        self.listed_val_data = [lmdb_patches[wsi.split(".")[0]] for wsi in self.listed_val_data if wsi.split(".")[0] in lmdb_patches]
        self.listed_val_labels = [wsi_labels[1:].astype(np.int) for wsi_labels in validation_folds.to_numpy() if wsi_labels[0].split(".")[0] in lmdb_patches]

        print("train_data length: ", len(self.listed_train_data))
        print("val_data length: ", len(self.listed_val_data))

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lmdb_patches_path', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--wsi_to_diagnosis_path', type=str, required=True, help='path to the wsi -> reports')
        parser.add_argument('--listed_data_path', type=str, required=True, help='path to the data partition we want to use')
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--set_size', type=int, default=32, help='size of the set for each WSI')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=True, help='whether to use shuffling during sampling')
        parser.add_argument('--train_folds', nargs='+', help='<Required> Set flag', required=True)

        return parser

    def setup(self, stage=None):
        self.train_dataset = TextImageDataset(self.lmdb_patches_path,
                                              self.wsi_to_diagnosis,
                                              self.listed_train_data,
                                              image_size=self.image_size,
                                              shuffle=self.shuffle,
                                              set_size=self.set_size,
                                              training_phase=True)

        self.val_dataset = TextImageDataset(self.lmdb_patches_path,
                                              self.wsi_to_diagnosis,
                                              self.listed_val_data,
                                              image_size=self.image_size,
                                              shuffle=False,
                                              set_size=self.set_size,
                                              validation_labels=self.listed_val_labels,
                                              training_phase=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False, collate_fn=self.dl_collate_fn)

    def dl_collate_fn(self, batch):
        image_tensor = torch.stack([torch.stack(row[0]) for row in batch])
        if len(batch[0])==3:
            validation_labels = torch.stack([torch.tensor(row[2]) for row in  batch])
            wsi_names = [row[1] for row in batch]
            return image_tensor, wsi_names, validation_labels

        # when validating the text tensor is a tensor with strings for the wsi_names
        text_tensor = torch.stack([row[1] for row in batch])

        return image_tensor, text_tensor
