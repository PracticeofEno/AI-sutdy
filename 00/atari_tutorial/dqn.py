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

steps_done = 0


class CNN(nn.Module):

    def __init__(self, n_observations, n_actions, env):
        super(CNN, self).__init__()
        self.keep_porb = 0.5
        self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d
		)
    def forward(self, x):
    
    def select_action(self, state):
        