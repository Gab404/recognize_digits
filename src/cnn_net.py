import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(6, 16, 5)
        self.linear_layer1 = nn.Linear(16 * 16, 120)
        self.linear_layer2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv_layer2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 16)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        return x