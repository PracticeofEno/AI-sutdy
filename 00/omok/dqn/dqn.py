import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

steps_done = 0

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, env):
        super(DQN, self).__init__()
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the AdamW optimizer
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.stps_done = 0
        self.env = env
        self.memory = ReplayMemory(10000)
        
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 128)
        self.layer6 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)
    
    def select_action(self, state):
        
        global steps_done
        sample = random.random()
        
        # 에피소드를 진행할수록 랜덤하게 뽑지않고 최적값을 기준으로 액션을 하게 만들기 위한 변수
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        
        if sample > eps_threshold:
            with torch.no_grad():
                # 현재 가능한 액션중에서 바둑알이 놓여진곳을 제외한 곳중 가장 큰 q값을 갖는놈자리 선택
                board_tensor = torch.tensor(self.env.board, device=device, dtype=torch.long)
                mask = (board_tensor == 0)
                x_flatten = mask.flatten()
                true_indices = torch.nonzero(x_flatten).flatten()
                board_tensor = self(state).flatten()
                selected_values = torch.gather(board_tensor, 0, true_indices)
                indicate = torch.nonzero(board_tensor == selected_values.max(0)[0])
                return torch.tensor([[indicate]], device=device, dtype=torch.long)
        else:
            return torch.tensor([[self.env.sample()]], device=device, dtype=torch.long)