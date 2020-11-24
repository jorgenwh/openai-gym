from qnetwork import QNetwork
from memory import MemoryReplay, Transition
import numpy as np
import torch
import os

class Agent:
    def __init__(self, gamma, epsilon, ep_min, ep_decay, lr, 
            batch_size, n_actions, mem_size, cuda):
        self.gamma = gamma
        self.epsilon = epsilon
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.batch_size = batch_size

        self.memory = MemoryReplay(mem_size=mem_size)
        self.q_network = QNetwork(lr, 160, 160, self.n_actions, cuda)

    def remember(self, *args):
        self.memory.add_memory(*args)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        state = torch.Tensor([state]).to(self.q_network.device)
        with torch.no_grad():
            actions = self.q_network(state)
        return torch.argmax(actions).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.get_batch(self.batch_size)
        batch = Transition(*zip(*batch))
        
        state_batch = torch.Tensor(batch.state).to(self.q_network.device)
        next_state_batch = torch.Tensor(batch.next_state).to(self.q_network.device)
        reward_batch = torch.cat(batch.reward).to(self.q_network.device)

        q_eval = self.q_network(state_batch)
        
        next_q_eval = self.q_network(next_state_batch)

        target_q_value = reward_batch + self.gamma * torch.max(next_q_eval)
        target_q_value[batch.terminal] = 0.0

        target_q_eval = q_eval.clone()
        target_q_eval[0][batch.action] = target_q_value

        self.q_network.optimizer.zero_grad()
        loss = self.q_network.loss_function(target_q_eval, q_eval)
        loss.backward()
        self.q_network.optimizer.step()

        self.epsilon = max(self.epsilon - self.ep_decay, self.ep_min)

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
