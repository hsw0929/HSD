import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import numpy as np
from os.path import isfile, join

class DVSCIFAR10(Dataset):
    def __init__(self, data_path='/data/users/husw/datasets/cifar10-dvs/frames_number_10_split_by_number',
                 data_type='train', transform=False):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(10)
        self.transform = transform
        self.cut=transforms.RandomCrop(size=48,padding=4)
        self.horfilp=transforms.RandomHorizontalFlip()
        self.rotate = transforms.RandomRotation(degrees=15)

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        if self.transform:

            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        return data, label

    def __len__(self):
        return self.data_num



