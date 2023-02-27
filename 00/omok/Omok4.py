import torch
from rule import *
import random
import numpy as np


class Omok(object):
    def __init__(self, board_size):
        self.board_size = board_size
        self.actions = 2
        self.board = [[0 for i in range(self.board_size)] for j in range(self.board_size)]
        self.rule = Rule(self.board, self.board_size)
        self.turn = 1
        self.init_board()
        self.id = 1
        self.is_gameover = False
        self.winner = -1
        self.end = False
        

    def init_board(self):
        for y in range(self.board_size):
            for x in range(self.board_size):
                self.board[y][x] = 0

    def put_stone(self, x, y, stone):
        if self.rule.is_invalid(x,y) == True or self.board[x][y] != 0:
            return 0
        self.board[x][y] = stone
        cnt = self.is_bridged(x,y, stone)
        if cnt >= 5:
            self.winner = stone
            self.end = True
            if stone == 1:
                return 10
            else:
                return -10
            
        self.id += 1
        if (self.id == 100):
            self.end = True
            return 0
        return 0
    
    def print_board(self):
        i = 0
        for i in range(len(self.board)):
            print(self.board[i])
            i += 1
        print()
    
    def get_env(self):
        env = np.array(self.board).ravel()
        return env
    
    def sample(self):
        x = random.randint(0, self.board_size - 1)
        y = random.randint(0, self.board_size - 1)
        ret = (x * 10) + y
        while self.board[x][y] != 0 :
            x = random.randint(0, self.board_size - 1)
            y = random.randint(0, self.board_size - 1)
            ret = (x * 10) + y
        return ret
    
    def step(self, action):
        x = action // 10
        y = action % 10
        reward = self.put_stone(x, y, 1)
        
        tmp = self.sample()
        x = tmp // 10
        y = tmp % 10
        reward_tmp = self.put_stone(x, y, 2)
        if (reward_tmp == -10):
            reward = reward_tmp
        
        observation = np.array(self.board).ravel()
        
        terminated = self.end
        return observation, reward, terminated
    
    def reset(self):
        self.init_board()
        self.turn = 1
        self.id = 1
        self.is_gameover = False
        self.winner = -1
        self.end = False
        s = np.array(self.board).ravel()
        return s
    
    def is_bridged(self, x, y, stone):
        dx = [1, 0, 1, 1]
        dy = [0, 1, 1, -1]
        max_cnt = 0
        for i in range(4):
            cnt = 1
            cx, cy = x, y
            while (True):
                cx, cy = cx + dx[i], cy + dy[i]
                if self.rule.is_invalid(cx, cy) or self.board[cx][cy] != stone:
                    break
                else:
                    cnt += 1
            cx, cy = x, y
            while (True):
                cx, cy = cx - dx[i], cy - dy[i]
                if self.rule.is_invalid(cx, cy) or self.board[cx][cy] != stone:
                    break
                else:
                    cnt += 1
            max_cnt = max(cnt, max_cnt)
        return max_cnt