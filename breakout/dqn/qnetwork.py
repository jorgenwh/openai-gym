import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, lr, n_actions, cuda):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(22528, 1024)
        self.fc2 = nn.Linear(1024, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and cuda else 'cpu')
        self.to(self.device)

        print(self, "\n", self.device, sep="")

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        out = self.fc2(x)

        return out