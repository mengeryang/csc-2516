import os
import io
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class InceptionDataset(Dataset):

    def __init__(self, datasetType, datasetFile, split=["train", "test", "valid"]):
        self.datasetType = datasetType
        self.datasetFile = datasetFile
        self.dataset = None
        self.dataset_keys = None
        self.h5py2int = lambda x: int(np.array(x))
        self.split = split

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = []
        for split in self.split:
            self.dataset_keys += [split + '/' + str(k) for k in f[split].keys()]
        length = len(self.dataset_keys)
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = []
            for split in self.split:
                self.dataset_keys += [split + '/' + str(k) for k in self.dataset[split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[example_name]

        # pdb.set_trace()

        right_image = bytes(np.array(example['img']))
        right_class = example['class'][()]
        if self.datasetType == "flowers":
            right_class = int(right_class.split("_")[1]) - 1
        else:
            right_class = int(right_class.split(".")[0]) - 1
        right_embed = np.array(example['embeddings'], dtype=float)

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        # inception v3 input size
        right_image = right_image.resize((299, 299))

        right_image = self.validate_image(right_image)

        sample = {
                'right_images': torch.FloatTensor(right_image),
                'right_classes': torch.LongTensor([right_class]),
                'right_embed': torch.FloatTensor(right_embed)
                 }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)

        return sample

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((299, 299, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)

