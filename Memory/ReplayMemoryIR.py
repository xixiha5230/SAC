import random
import numpy as np
from .ReplayMemory import ReplayMemory


class ReplayMemoryIR(ReplayMemory):
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        img = []
        img_next = []
        ray = []
        ray_next = []
        for i in batch:
            img.append(i[0][0])
            ray.append(i[0][1])
            img_next.append(i[3][0])
            ray_next.append(i[3][1])
        _, action, reward, _, done = map(np.stack, zip(*batch))
        state = [img, ray]
        next_state = [img_next, ray_next]
        return state, action, reward, next_state, done
