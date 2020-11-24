import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, lr, h, w, outputs, cuda):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_h1 = self.conv_size_out(h, 8, 4)
        conv_w1 = self.conv_size_out(w, 8, 4)
        conv_h2 = self.conv_size_out(conv_h1, 4, 2)
        conv_w2 = self.conv_size_out(conv_w1, 4, 2)
        conv_h3 = self.conv_size_out(conv_h2, 3, 1)
        conv_w3 = self.conv_size_out(conv_w2, 3, 1)

        self.fc1 = nn.Linear(conv_h3 * conv_w3 * 64, 1024)
        self.fc2 = nn.Linear(1024, outputs)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.loss_function = nn.MSELoss()
        self.loss_function = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and cuda else 'cpu')
        self.to(self.device)

        print(self, "\nDevice: ", self.device, sep="")

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out

    def conv_size_out(self, size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1