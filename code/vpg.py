import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import gym
from collections import deque

class network(nn.Module):
    def __init__(self, state_size, hidden_count, action_size):
        super(network, self).__init__()
        self.h = nn.Linear(state_size, hidden_count)
        self.out = nn.Linear(hidden_count, action_size)

    def pass(self, a):
        a = F.relu(self.h(a))
        a = F.softmax(self.out(a), dim=1)
        return a

