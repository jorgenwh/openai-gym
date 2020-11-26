import random
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))

class MemoryReplay:
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memories = []
        self.cntr = 0

    def add_memory(self, *args):
        if len(self.memories) < self.mem_size:
            self.memories.append(None)

        self.memories[self.cntr] = Transition(*args)
        self.cntr = (self.cntr + 1) % self.mem_size
    
    def get_batch(self, size):
        return random.sample(self.memories, size)

    def __len__(self):
        return (self.cntr + 1) % self.mem_size