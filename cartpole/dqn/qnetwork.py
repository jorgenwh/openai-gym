import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, lr, in_features, fc1, fc2, n_actions):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(in_features, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions