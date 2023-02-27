import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch

learning_rate = 0.0002
gamma = 0.98 

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)
    
    def forward(self, x):
        x =  F.relu(self.fc1(x))
        x =  F.softmax(self.fc2(x), dim=0)
        return x
    
    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R = 0
        for r, prob in self.data[::-1]: # 왜 뒤에서부터 ? 그래야 리턴이 계산됨. 맨 마지막꺼까지 더해야 되는데 3부터 세라고 하면 3부터 끝까지 세야 되서 뒤에서부터 
            R = r + gamma * R # 끝에서 부터 R 곱하기 감마 + 현재리워드 이런식으로 되서 R(t) + gamma * R(t+1)가 성립
            loss = -torch.log(prob) * R # log파이(At|St) * Gt * -1  -1은 기본적으로 경사하강법을 실행해서 - 를 붙여서 경사상승으로 만들어야함
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.data = []