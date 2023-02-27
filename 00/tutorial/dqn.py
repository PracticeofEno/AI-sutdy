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
        self.env = env
        self.memory = ReplayMemory(10000)
        
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def select_action(self, state):
        
        global steps_done
        sample = random.random()
        
        # 에피소드를 진행할수록 랜덤하게 뽑지않고 최적값을 기준으로 액션을 하게 만들기 위한 변수
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) 은 두 번째 차원(dimension)을 따라 최대값을 찾는 함수입니다. 
                # 여기서 두 번째 차원은 일반적으로 모델이 다중 클래스 분류(multi-class classification)를 수행하는 경우에 해당합니다. 
                # 따라서 이 함수는 모델이 예측한 각 클래스에 대한 예측 확률(probability) 중에서 가장 높은 확률을 갖는 클래스의 인덱스를 찾습니다
                # [1]은 반환된 최대값(maximum value)을 나타내는 텐서(tensor)입니다. 
                # 이 텐서는 모델이 예측한 각 클래스에 대한 예측 확률(probability) 중에서 가장 높은 확률을 갖는 클래스의 확률값
            	#  view(1,1)은 이 값을 1x1 크기의 텐서로 변환합니다. 이 작업은 모델이 다른 코드에서 기대하는 형태의 출력값을 생성하는 것을 보장하기 위한 것
             return self(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)