import torch.nn as nn 
import torch
from torchsummary import summary

class DualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer12 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer22 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10912, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 5)
    
    def forward(self, x, y ):
        out1 = self.layer1(x)
        out1 = self.layer2(out1)
        out1 = self.flatten(out1)

        out2 = self.layer12(y)
        out2 = self.layer22(out2)
        out2 = self.flatten(out2)

        out = torch.cat((out1, out2), 1) 

        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def architecture(modelName, dimension):
    lenet = modelName()
    summary(lenet, dimension)

# architecture(DualModel, [(1, 100, 100), (1, 100, 50)])

