import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ucb:
    '''
    Inputs:
        a: number of arms
        c: (hyperparam) constant (> 0)
        length: iterations
    Notes: 
        We randomize the reward distributions' means.
        The n-th reward (for n from 0 to a-1)
        has Gaussian distribution with mean sampled from N(0, 1) and variance 1. 
    '''
    def __init__ (self, a, c, length)
        self.arms = a
        self.c = c
        self.length = length
        # total number of samples, set to 1 for no div 0
        self.curr = 1
        # number of times each reward was sampled
        self.rew_count = np.ones(a)

        self.mean_reward = 0
        self.rewards = np.zeros(length)
        self.mean_arms_reward = np.zeros(a)

        self.env = np.random.normal(0, 1, a)

    def select(self):
        A_rightexp = np.sqrt(np.log(self.curr)/self.rew_count)
        A = np.argmax(self.mean_reward + self.c * a_rightexp)
        reward = np.random.normal(self.env[A], 1)

        self.curr += 1
        self.rew_count[A] += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.curr 
        self.mean_arms_reward[A] = self.mean_arms_reward[A] + (reward - self.mean_arms_reward) / self.rew_count[A]

    def run(self):
        for i in range(self.length):
            self.select()
            self.rewards[i] = self.mean_reward

    def reset(self):
        self.curr = 1
        self.rew_count = np.ones(self.arms)
        self.mean_reward = 0
        self.rewards = np.zeros(self.length)
        self.mean_arms_reward = np.zeros(self.arms)
        self.env = np.linspace(0, a-1, a)



