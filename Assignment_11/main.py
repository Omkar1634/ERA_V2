import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms, datasets

from utils import plot_misclassified_images_optimized



class CIFAR10Dataset(datasets.CIFAR10):

 def __init__(self,root="./data",train=True,download=True,transform=None):
   super().__init__(root=root,train=train,download=download,transform=transform)

 def __getitem__(self,index):
   image, label = self.data[index], self.targets[index]

   if self.transform is not None:
     transformed = self.transform(image=image)
     image = transformed["image"]
   return image,label



class S_11:
    def __init__(self, model,classes, train_transforms, test_transforms, epochs = 20, optimizer_type='adam',scheduler_type = "step", use_scheduler=True):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_transforms =train_transforms
        self.test_transforms = test_transforms
        self.classes = classes
        #self.lr_max = lr_max
        self.epochs = epochs
        #self.max_at_epoch = max_at_epoch
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.misclassified_images = []
        self.misclassified_true_labels = []
        self.misclassified_pred_labels = []
        #self.optimizer =  optim.Adam(model.parameters(), lr=lr_max)
        # self.scheduler =   optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr_max,
                                                       #epochs=epochs, steps_per_epoch=len(train_loader),
                                                       #div_factor=lr_max/lr_min, final_div_factor=lr_max/lr_min,
                                                       #pct_start=max_at_epoch/epochs)
        # Initialize optimizer based on user input
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Initialize scheduler if requested
        if use_scheduler:
            if scheduler_type.lower() == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            elif scheduler_type.lower() == 'multistep':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
            elif scheduler_type.lower() == 'exponential':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
            elif scheduler_type.lower() == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        else:
            self.scheduler = None  # No scheduler if not requested
   

    def split_data(self):
        print(" ==> Preparing data... ")

        SEED = 50
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        # dataloader arguments
        dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=64)

        # train dataloader
        train = CIFAR10Dataset(root="./data",download=True,transform = self.train_transforms)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args) 

        # test dataloader
        test = CIFAR10Dataset(root="./data",download=True,train=False,transform = self.test_transforms)
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

        return train_loader,test_loader
        
    def get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()


    def train(self,train_loader):
        self.model.train()
        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            correct = self.get_correct_pred_count(pred, target)
            pbar.set_description(desc=f'Loss={loss.item():0.4f} Accuracy={100 * correct / len(data):0.2f}')
            self.train_losses.append(loss.item())
            self.train_acc.append(100 * correct / len(data))

    def test(self,test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                correct += self.get_correct_pred_count(output, target)

                # Collect misclassified images
                predicted_labels = output.argmax(dim=1)
                misclassified_mask = predicted_labels != target
                self.misclassified_images.extend(data[misclassified_mask].cpu().numpy())
                self.misclassified_true_labels.extend(target[misclassified_mask].cpu().numpy())
                self.misclassified_pred_labels.extend(predicted_labels[misclassified_mask].cpu().numpy())
                

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
        self.test_acc.append(100. * correct / len(test_loader.dataset))

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
    
    def plot_acc_loss(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
        plt.show()
 
    def run(self):
        train_loader, test_loader = self.split_data() # Get train_loader and test_loader
        #misclassified_images, misclassified_true, misclassified_pred = self.test()  # Get misclassified_images, misclassified_true, misclassified_pred
        print("==> Satrting Training & Testing")
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}/{self.epochs}')
            self.train(train_loader)  # Pass train_loader to train method
            self.test(test_loader)    # Pass test_loader to test method
        self.plot_acc_loss()
        #plot_misclassified_images(self.misclassified_images, self.misclassified_true_labels, self.misclassified_pred_labels, self.classes)
        




