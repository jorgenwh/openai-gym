import numpy as np
import torch

from policy_network import PolicyNetwork

class Agent:
    def __init__(self, cuda):
        self.action_memory = []
        self.reward_memory = []
        self.discount = 0.99

        self.pi = PolicyNetwork(lr=0.001, cuda=cuda)


    def act(self, observation):
        state = torch.Tensor([observation]).to(self.pi.device)

        out = self.pi(state)
        probabilities = torch.softmax(out, dim=1)

        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        action_log_prob = action_probs.log_prob(action)

        self.action_memory.append(action_log_prob)

        return action.item()

    def remember(self, action_probs, reward):
        self.action_memory.append(action_probs)
        self.reward_memory.append(reward)

    def learn(self):
        self.pi.optimizer.zero_grad()

        R = []
        for t in range(len(self.reward_memory)):
            G = 0
            gamma = 1
            for k in range(t, len(self.reward_memory)):
                G += self.reward_memory[k] * gamma
                gamma *= self.discount
            R.append(G)

        loss = 0
        for action_log_prob, reward in zip(self.action_memory, R):
            loss += -reward * action_log_prob
        
        loss.backward()
        self.pi.optimizer.step()

        self.action_memory.clear()
        self.reward_memory.clear()

    