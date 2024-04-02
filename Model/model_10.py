import torch
import  torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  
        out = self.relu(out)
        return out

class Custom_Resnet(nn.Module):
    def __init__(self):
        super(Custom_Resnet, self).__init__()

        # PrepLayer

        self.prelayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        self.layer1_X = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer1_R1 = ResBlock(128, 128)

        # Layer 2
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1, padding=1,bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        self.layer3_X = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer3_R3 = ResBlock(512,512)

        # Maxpooling layer
        self.maxpooling = nn.MaxPool2d(4,4)
        # Fully connected layer
        self.fc = nn.Linear(512,10)



    def forward(self, x):
        x = self.prelayer(x)
        X = self.layer1_X(x)
        R1 = self.layer1_R1(X)
        out = X + R1 
        out = self.layer_2(out)
        X2 = self.layer3_X(out)
        R2 = self.layer3_R3(X2)
        out_1 = X2 + R2
        out_2 = self.maxpooling(out_1)
        out_2 = out_2.view(out_2.size(0),-1)
        final = self.fc(out_2)
        return final




        
def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)
