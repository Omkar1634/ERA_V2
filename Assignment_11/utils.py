import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std





train_transforms = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    A.CoarseDropout(max_holes = 16,max_height=16,max_width=16,min_holes = 16,min_height=16,min_width=16,fill_value=(0.49139968,0.48215841,0.44653091), mask_fill_value = None),
    A.Normalize(mean=(0.49139968,0.48215841,0.44653091), std=(0.24703223,0.24348513,0.26158784)),
    ToTensorV2()
                            ])


test_transforms = A.Compose([ A.Normalize(mean=(0.49139968,0.48215841,0.44653091), std=(0.24703223,0.24348513,0.26158784)),
                            ToTensorV2()
                                    ])


