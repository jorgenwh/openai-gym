from qnetwork import QNetwork
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
        self.action_space = [i for i in range(n_actions)]
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.empty((self.mem_size, 4))
        self.action_memory = np.empty(self.mem_size, dtype=np.int32)
        self.reward_memory = np.empty(self.mem_size)
        self.next_state_memory = np.empty((self.mem_size, 4))
        self.done_memory = np.empty(self.mem_size, dtype=np.bool)

        self.q_network = QNetwork(lr, 256, 256, n_actions, cuda)

    def remember(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        self.mem_cntr += 1

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        state = torch.Tensor([state]).to(self.q_network.device)
        actions = self.q_network(state)
        return torch.argmax(actions).item()

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.q_network.train()
        self.q_network.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size)

        state_batch = torch.Tensor(self.state_memory[batch]).to(self.q_network.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.Tensor(self.reward_memory[batch]).to(self.q_network.device)
        next_state_batch = torch.Tensor(self.next_state_memory[batch]).to(self.q_network.device)
        done_batch = torch.Tensor(self.done_memory[batch]).type(torch.BoolTensor).to(self.q_network.device)

        q_eval = self.q_network(state_batch)[batch_index, action_batch]
        q_eval_next = self.q_network(next_state_batch)
        q_eval_next[done_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_eval_next, dim=1)[0]

        loss = self.q_network.loss_function(q_target, q_eval).to(self.q_network.device)
        loss.backward()
        self.q_network.optimizer.step()

        self.epsilon = max(self.epsilon - self.ep_decay, self.ep_min)
        self.q_network.eval()

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