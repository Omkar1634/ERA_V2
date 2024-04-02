import torch
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np


# Train Phase transformations
train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=None, mask_value=None),
                              A.RandomCrop(height=32, width=32, always_apply=True),
                              A.CoarseDropout(max_holes = 8,max_height=8,max_width=8,min_holes = 8,min_height=8,min_width=8,fill_value=(0.49139968,0.48215841,0.44653091), mask_fill_value = None),
                              A.Normalize(mean=(0.49139968,0.48215841,0.44653091), std=(0.24703223,0.24348513,0.26158784)),
                              ToTensorV2()])

# Test Phase transformations
test_transforms = A.Compose([
                                       A.Normalize(mean=(0.49139968,0.48215841,0.44653091), std=(0.24703223,0.24348513,0.26158784)),
                                       ToTensorV2()
                                       ])

# dataset for CIFAR10 dataset
class CIFAR10Dataset(datasets.CIFAR10):

 def __init__(self,root="./data",train=True,download=True,transform=None):
   super().__init__(root=root,train=train,download=download,transform=transform)

 def __getitem__(self,index):
   image, label = self.data[index], self.targets[index]

   if self.transform is not None:
     transformed = self.transform(image=image)
     image = transformed["image"]
   return image,label


def load_dataset():
    
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
     torch.cuda.manual_seed(SEED)

    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train = CIFAR10Dataset(transform = train_transforms)
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args) 

    # test dataloader
    test = CIFAR10Dataset(transform = test_transforms,train=False)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args) 

    return train_loader,test_loader