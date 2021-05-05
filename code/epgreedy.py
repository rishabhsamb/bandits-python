import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random
import Math

class eps_greedy:
    '''
    Inputs:
        a: number of arms
        eps: the epsilon probability for which non-greedy actions will be chosen (hyperparam)
        length: iterations
    Notes: 
        We randomize the reward distributions' means.
        The n-th reward (for n from 0 to a-1)
        has Gaussian distribution with mean sampled from N(0, 1) and variance 1. 
    '''
    def __init__(self, a, eps, length):
        self.arms = a
        self.eps = eps
        self.length = length

        self.curr = 1 # current number of total samples, set to 1 for no div0 
        self.rew_count = np.ones(a) # current number of samples for each action 
        self.mean_reward = 0 # current mean reward over all samples 
        self.rewards = np.zeros(length) # vector of mean reward at each iteration  
        self.mean_arms_reward = np.zeros(a) + a # optimistic initial values 

        self.env = np.random.normal(0, 1, a) # See notes.

    def random_nonmax_index(self, max_ind):
        ind = Math.floor(random.uniform(0, 1) * (self.arms))
        if max_ind == ind:
            return self.random_nonmax_index(self, max_ind)
        else:
            return ind 

    def select(self):
        greedy = np.argmax(self.mean_arms_reward)
        rand = random.uniform(0, 1)
        
        if rand <= self.eps:
            A = self.random_nonmax_index(self, greedy)
        else:
            A = greedy

        reward = np.random.normal(self.env[A], 1)

        self.curr += 1
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.curr 
        self.rew_count[A] += 1
        self.mean_arms_reward[A] = self.mean_arms_reward[A] + (reward - self.mean_arms_reward) / self.rew_count[A]

    def reset(self):
        self.curr = 1
        self.rew_count = np.ones(self.arms)
        self.mean_reward = 0
        self.rewards = np.zeros(self.length)
        self.mean_arms_reward = np.zeros(self.arms) + self.arms 
        self.env = np.linspace(0, a-1, a)
