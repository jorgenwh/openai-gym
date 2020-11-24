from policy_network import PolicyNetwork
import numpy as np
import torch
import torch.nn.functional as F
import os

class Agent:
    def __init__(self, n_actions, lr=0.001, gamma=0.99, cuda=True):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(lr, n_actions, cuda)

    def act(self, observation):
        observation = torch.Tensor(observation)
        probabilities = F.softmax(self.policy(observation))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def push_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        # calculate all returns
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # normalize the returns
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = torch.Tensor(G, dtype=torch.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory.clear()
        self.reward_memory.clear()

    def save_model(self, name):
        folder = "models/"
        if not os.path.exists(folder):
            print(f"Cannot find folder '{folder}' when trying to save model.")
            return

        filename = folder + name
        i = 1
        while os.path.isfile(filename):
            filename = folder + name + str(i)
            i += 1

        torch.save(self.q_network.state_dict(), filename)

    def load_model(self, name):
        folder = "models/"
        if not (os.path.exists(folder) or os.path.isfile(folder + name)):
            raise FileNotFoundError(f"Cannot find model '{folder + name}' when trying to load model.")

        self.q_network.load_state_dict(torch.load(folder + name))
        self.epsilon = self.ep_min = 0.0