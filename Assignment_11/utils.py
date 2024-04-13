import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt




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



classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=None, mask_value=None),
    A.RandomCrop(height=32, width=32, always_apply=True),
    A.CoarseDropout(max_holes=16, max_height=16, max_width=16, min_holes=16, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215841, 0.44653091), mask_fill_value=None),
    A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
    ToTensorV2()
])


def plot_misclassified_images_optimized(images, true_labels, predicted_labels, classes, max_images=10):
    # Ensure images are in the correct shape and range for plotting
    images = [img.transpose((1, 2, 0)) for img in images[:max_images]]
    true_labels = true_labels[:max_images]
    predicted_labels = predicted_labels[:max_images]
    
    fig, axes = plt.subplots((len(images) + 1) // 2, 2, figsize=(10, 20))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = np.clip(images[i], 0, 1)  # Assuming images were normalized beforehand
            ax.imshow(img)
            ax.set_title(f"True: {classes[true_labels[i].item()]}, Pred: {classes[predicted_labels[i].item()]}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

