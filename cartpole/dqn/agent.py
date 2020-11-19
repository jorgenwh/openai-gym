from qnetwork import QNetwork
from memory import Memory
import numpy as np
import torch

class Agent:
    def __init__(self, gamma, epsilon, lr, in_features, batch_size, n_actions, 
            mem_size=100_000, ep_min=0.01, ep_decay=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.memory = Memory(mem_size)

        self.q_network = QNetwork(lr, in_features, 256, 256, n_actions)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        state = torch.Tensor([state]).to(self.q_network.device)
        actions = self.q_network(state)
        return torch.argmax(actions).item()

    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        
        self.q_network.train()
        self.q_network.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.memory.get_batch(self.batch_size)

        state_batch = torch.Tensor(states).to(self.q_network.device)
        action_batch = torch.Tensor(actions).type(torch.long)
        reward_batch = torch.Tensor(rewards).to(self.q_network.device)
        next_state_batch = torch.Tensor(next_states).to(self.q_network.device)
        done_batch = torch.Tensor(dones).type(torch.bool).to(self.q_network.device)

        q_next = self.q_network(next_state_batch)
        q_targets = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        q_next[done_batch] = 0.0
        q_next[action_batch.reshape(self.batch_size, 1)] = q_targets
        q_eval = self.q_network(state_batch)

        loss = self.q_network.loss_function(q_targets, q_eval).to(self.q_network.device)
        loss.backward()
        self.q_network.optimizer.step()

        self.epsilon = max(self.epsilon - self.ep_decay, self.ep_min)
        self.q_network.eval()
