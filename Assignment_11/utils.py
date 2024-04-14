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



class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']
    

train_transforms = AlbumentationsTransform(A.Compose([
    A.RandomCrop(height=32, width=32, p=0.2,always_apply=True),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, fill_value=[0.49139968*255, 0.48215827*255 ,0.44653124*255], mask_fill_value=None),
    A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
    ToTensorV2()
]))

test_transforms = AlbumentationsTransform(A.Compose([
    A.Normalize(mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)),
    ToTensorV2()
]))


def find_and_visualize_misclassified_images(model, device, test_loader, criterion, classes, num_images=10):
    model.eval()  # Set the model to evaluation mode
    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            misclassified_idxs = (preds != target).nonzero(as_tuple=False).squeeze()

            for idx in misclassified_idxs:
                if len(misclassified_images) < num_images:
                    misclassified_images.append(data[idx].cpu())
                    misclassified_true.append(target[idx].cpu())
                    misclassified_pred.append(preds[idx].cpu())
                else:
                    plot_misclassified_images(misclassified_images, misclassified_true, misclassified_pred, classes)
                    return
    if misclassified_images:
        plot_misclassified_images(misclassified_images, misclassified_true, misclassified_pred, classes)
import matplotlib.pyplot as plt

def plot_misclassified_images(images, true_labels, predicted_labels, classes):
    fig, axes = plt.subplots((len(images) + 1) // 2, 2, figsize=(8,8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].numpy().transpose((1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            ax.imshow(img)
            ax.set_title(f"True: {classes[true_labels[i].item()]}, Pred: {classes[predicted_labels[i].item()]}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

