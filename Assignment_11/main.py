import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms, datasets
import numpy as np
from torch_lr_finder import LRFinder


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']

class S_11:
    def __init__(self, model, classes, train_transforms, test_transforms, epochs=20, optimizer_type='sgd', scheduler_type="step", use_scheduler=True):
        self.model = model
        self.classes = classes
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.use_scheduler = use_scheduler
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        # Define the device as CUDA if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move the model to the specified device
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # Placeholder for the optimizer and scheduler, will be initialized after LR finding
        self.optimizer = None
        self.scheduler = None

    def find_optimal_lr(self, train_loader):
        # Placeholder learning rate for LR Finder initialization
        placeholder_lr = 1e-7
        temp_optimizer = torch.optim.SGD(self.model.parameters(), lr=placeholder_lr, momentum=0.9, weight_decay=1e-4)
        
        lr_finder = LRFinder(self.model, temp_optimizer, self.criterion, device='cuda')
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.plot()
        
        # Find the optimal learning rate
        self.learning_rate = self._find_optimal_lr(lr_finder)
        lr_finder.reset()
        
        # Initialize the optimizer with the optimal learning rate
        self._init_optimizer()

        # Initialize the scheduler
        self._init_scheduler()

    def _find_optimal_lr(self, lr_finder):
        # Assuming lr_finder has methods to provide the logged learning rates and losses
        lrs = np.array(lr_finder.history['lr'])
        losses = np.array(lr_finder.history['loss'])
        # Calculate the gradient of the losses
        gradients = np.gradient(losses)
        # Find the steepest gradient
        steepest_slope_index = np.argmin(gradients)
        return lrs[steepest_slope_index]

    def _init_optimizer(self):
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

    def _init_scheduler(self):
        if self.use_scheduler:
            if self.scheduler_type.lower() == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            elif self.scheduler_type.lower() == 'multistep':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
            elif self.scheduler_type.lower() == 'exponential':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
            elif self.scheduler_type.lower() == 'plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
            elif self.scheduler_type.lower() == 'onecycle':
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.learning_rate,steps_per_epoch = 98,
                                                epochs=self.epochs,
                                                pct_start=int(0.3*self.epochs)/self.epochs if self.epochs != 1 else 0.5,   # 30% of total number of Epochs
                                                div_factor=100,
                                                three_phase=False,
                                                final_div_factor=100,
                                                anneal_strategy="linear")
    

    def split_data(self):
        print(" ==> Preparing data... ")

        SEED = 50
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        # dataloader arguments
        dataloader_args = dict(shuffle=True, batch_size=512, num_workers=2, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=64)

        # train dataloader
        train = datasets.CIFAR10(root="./data",download=True,transform = self.train_transforms)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args) 

        # test dataloader
        test = datasets.CIFAR10(root="./data",download=True,train=False,transform = self.test_transforms)
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
        train_loader, test_loader = self.split_data()
        
        # Find and set the optimal learning rate before the main training loop
        self.find_optimal_lr(train_loader)
        
        print("==> Starting Training & Testing")
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}/{self.epochs}')
            self.train(train_loader)
            self.test(test_loader)
        print('Finished Training. Saving the model...')
        self.save_model('trained_model.pth')
        print("Model saved!!")
        self.plot_acc_loss()
        
    def save_model(self, file_path='model.pth'):
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')



