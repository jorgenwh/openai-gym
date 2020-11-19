from collections import deque
import numpy as np
import random

class Memory:
    def __init__(self, mem_size):
        self.memories = deque(maxlen=mem_size)

    def add_memory(self, state, action, reward, next_state, done):
        self.memories.append((state, action, reward, next_state, done))

    def get_batch(self, size):
        batch = random.choices(self.memories, k=size)

        states = [batch[i][0] for i in range(size)]
        actions = [batch[i][1] for i in range(size)]
        rewards = [batch[i][2] for i in range(size)]
        next_states = [batch[i][3] for i in range(size)]
        dones = [batch[i][4] for i in range(size)]

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.memories)