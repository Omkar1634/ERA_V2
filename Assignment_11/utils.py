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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_pil_image


class MisclassificationVisualizer:
    def __init__(self, model, device, test_loader, classes):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.classes = classes

    def find_and_visualize_misclassified_images(self, num_images=10):
        self.model.eval()  # Set the model to evaluation mode
        misclassified_images = []
        misclassified_true = []
        misclassified_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, preds = torch.max(output, 1)
                misclassified_idxs = (preds != target).nonzero(as_tuple=False).squeeze()

                for idx in misclassified_idxs:
                    if len(misclassified_images) < num_images:
                        misclassified_images.append(data[idx].cpu())
                        misclassified_true.append(target[idx].cpu())
                        misclassified_pred.append(preds[idx].cpu())
                    else:
                        self.plot_misclassified_images(misclassified_images, misclassified_true, misclassified_pred)
                        return
        if misclassified_images:
            self.plot_misclassified_images(misclassified_images, misclassified_true, misclassified_pred)

    def plot_misclassified_images(self, images, true_labels, predicted_labels):
        fig, axes = plt.subplots((len(images) + 1) // 2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                img = images[i].numpy().transpose((1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
                ax.imshow(img)
                ax.set_title(f"True: {self.classes[true_labels[i].item()]}, Pred: {self.classes[predicted_labels[i].item()]}")
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.show()





class GradCamVisualizer:
    def __init__(self, model, device, test_loader, classes):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.classes = classes
        self.model.eval()

    def find_misclassified_images(self, num_images=10):
        misclassified_images = []
        misclassified_preds = []
        misclassified_true = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, preds = torch.max(output, 1)
                misclassified_idxs = (preds != target).nonzero(as_tuple=False).squeeze()
                for idx in misclassified_idxs:
                    if len(misclassified_images) < num_images:
                        misclassified_images.append(data[idx])
                        misclassified_true.append(target[idx])
                        misclassified_preds.append(preds[idx])
                    else:
                        return misclassified_images, misclassified_preds, misclassified_true
        return misclassified_images, misclassified_preds, misclassified_true

    def apply_gradcam(self, misclassified_images, misclassified_preds, target_layer):
        # Adjusting this line according to the new API
        cam = GradCAM(model=self.model, target_layers=[target_layer])
        for i, (img_tensor, pred) in enumerate(zip(misclassified_images, misclassified_preds)):
            input_tensor = img_tensor.unsqueeze(0)
            # Create an object for the target category based on the predicted class
            targets = [ClassifierOutputTarget(pred.item())]
            
            # Generate the CAM mask for the target category
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Visualize
            img = np.array(to_pil_image(img_tensor))
            grayscale_cam_uint8 = np.uint8(255 * grayscale_cam)

            # Now use this for visualization
            cam_img = show_cam_on_image(img / 255., grayscale_cam_uint8, use_rgb=True)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Image {i+1}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cam_img)
            plt.title(f"Grad-CAM {i+1}")
            plt.axis('off')

            plt.show()

    def visualize(self, target_layer, num_images=10):
        misclassified_images, misclassified_preds, misclassified_true = self.find_misclassified_images(num_images)
        self.apply_gradcam(misclassified_images, misclassified_preds, target_layer)





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