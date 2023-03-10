import gym
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from Policy import Policy

learning_rate = 0.0002
gamma = 0.98 


def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
        s = env.reset()
        s = torch.tensor(s[0], dtype=torch.float32)
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(s)
            m = Categorical(prob)
            a = m.sample()
            tmp = env.step(a.item())
            s_prime = torch.tensor(tmp[0], dtype=torch.float32)
            r = tmp[1]
            done = tmp[2]
            
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()
            