import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, n_outputs, cuda):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and cuda else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out