import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, train_loader, test_loader, lr_min, lr_max, epochs, max_at_epoch):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs = epochs
        self.max_at_epoch = max_at_epoch
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.optimizer = optim.Adam(model.parameters(), lr=lr_max)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr_max,
                                                       epochs=epochs, steps_per_epoch=len(train_loader),
                                                       div_factor=lr_max/lr_min, final_div_factor=lr_max/lr_min,
                                                       pct_start=max_at_epoch/epochs)

    def get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            correct = self.get_correct_pred_count(pred, target)
            pbar.set_description(desc=f'Loss={loss.item():0.4f} Accuracy={100 * correct / len(data):0.2f}')
            self.train_losses.append(loss.item())
            self.train_acc.append(100 * correct / len(data))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                correct += self.get_correct_pred_count(output, target)

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({100. * correct / len(self.test_loader.dataset):.2f}%)')

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
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}/{self.epochs}')
            self.train()
            self.test()
        self.plot_acc_loss()
