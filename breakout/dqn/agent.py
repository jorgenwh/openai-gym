from qnetwork import QNetwork
import numpy as np
import torch

class Agent:
    def __init__(self, gamma, epsilon, lr, batch_size, mem_size=100_000, 
            ep_min=0.01, ep_decay=5e-4, cuda=True):
        self.gamma = gamma
        self.epsilon = epsilon
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.n_actions = 4
        self.action_space = [i for i in range(self.n_actions)]
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.mem_cntr = 0

        self.state_memory = np.empty((self.mem_size, 3, 210, 160))
        self.action_memory = np.empty(self.mem_size, dtype=np.int32)
        self.reward_memory = np.empty(self.mem_size)
        self.next_state_memory = np.empty((self.mem_size, 3, 210, 160))
        self.done_memory = np.empty(self.mem_size, dtype=np.bool)

        self.q_network = QNetwork(lr, self.n_actions, cuda)

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
        torch.save(self.q_network.state_dict(), name)
