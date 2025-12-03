from collections import deque
import random

class memory_back:
    def __init__(self, max_mem):
        self.mem = deque(maxlen=max_mem)
        
    def add(self, experience):
        self.mem.append(experience)
        
    def add_batch(self, experiences):
        self.mem.extend(experiences)
        
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)