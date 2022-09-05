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
        action = []
        reward = []
        done = []
        for i in batch:
            img.append(i[0][0])
            ray.append(i[0][1])
            action.append(i[1])
            reward.append(i[2])
            img_next.append(i[3][0])
            ray_next.append(i[3][1])
            done.append(i[4])
            
        state = [img, ray]
        next_state = [img_next, ray_next]
        return state, np.array(action), np.array(reward), next_state,  np.array(done)
