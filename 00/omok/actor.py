import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

learning_rate = 0.0002
gamma         = 0.98

class ActorCritic(nn.Module):
    def __init__(self, device):
        super(ActorCritic, self).__init__()
        self.data = []
        self.n_rollout     = 500
        self.device = device
        
        self.fc1 = nn.Linear(100,256)
        self.fc2 = nn.Linear(256,256)
        self.fc_pi = nn.Linear(256,100)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            
            
        s_batch = torch.stack(s_lst, 0)
        a_batch = torch.tensor(a_lst, dtype=torch.int64, device=self.device)
        r_batch = torch.tensor(r_lst, dtype=torch.float, device=self.device) 
        s_prime_batch =  torch.stack(s_prime_lst, 0)
        done_batch = torch.tensor(done_lst, dtype=torch.float, device=self.device)
        
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = torch.gather(pi, 1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()  