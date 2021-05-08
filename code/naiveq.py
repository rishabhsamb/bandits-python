'''
Naive Q Learning is a very 'what you see is what you get' type algorithm,
as evidenced by the naive update function that is only tractable for relatively small state set cardinality.
'''

import gym 
Q = {}

g = 0.99
a = 0.3

env = gym.make("slime-volley")
action_iterator = range(env.action_space)

def Q_learn(s1, r, a, s2, pred):
    maxQ = max([Q[s2, a] for a in action_iterator])
    Q[s1, a] += a * (r + g * maxQ * (1 - pred) - Q[s1, a]) 
