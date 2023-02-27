from collections import namedtuple
from actor import ActorCritic
import torch.optim as optim
import torch
import torch.nn as nn

class Agent:
    def __init__(self, n_observations, n_actions, env, device, number):
        self.policy_net = ActorCritic()
        self.device = device
        self.number = number
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.policy_net.LR, amsgrad=True)
        
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        
    def optimize_model(self):
        if len(self.policy_net.data) < self.policy_net.n_rollout:
            return
        