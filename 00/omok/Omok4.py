import torch
from rule import *
import random
import numpy as np

PATTERNS = {
    '|OOOO_|': 10**9,
    '|OOO_O|': 10**9,
    '|OO_OO|': 10**9,

    '| OOO_ |': 10**7,
    '| OO_O |': 10**7,
    '|O O_O O|': 10**7,

    '|OOO _|': 10**5,
    '|OO O_|': 10**5,
    '|O OO_|': 10**5,
    '| OOO_|': 10**5 + 1,
    '|OOO_ |': 10**5 + 1,
    '|OO _O|': 10**5,
    '|O O_O|': 10**5,
    '| OO_O|': 10**5,
    '|OO_O |': 10**5,
    '|OO_ O|': 10**5,

    '| OO _ |': 10**3,
    '| O O_ |': 10**3,
    '|  OO_ |': 10**3 + 1,
    '| OO_  |': 10**3 + 1,
    '| O _O |': 10**3,
    '|  O_O |': 10**3 + 1,

    '|OO  _|': 10,
    '|O O _|': 10,
    '|OO _ |': 10,
    '|O O_ |': 10,
    '|O _ O|': 10,
    '|OO_  |': 11,

    '| O  _ |': 1,
    '|  O _ |': 1,
    '|   O_ |': 2,
    '| O _ |': 1,
    '|  O_ |': 2,
    '| O_  |': 2,
}

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
        
        self.tmp = 0
        self.ai_win = True
        

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
                self.ai_win = False
                return 1
            else:
                self.ai_win = True
                return -1
            
        self.id += 1
        if (self.id == 100):
            self.end = True
            return 0
        return cnt * 0.001
    
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
        
        
        
        if (self.ai_win):
            p  = random.uniform(0, 1)
            if p > 0.3 or self.tmp < 5000:
                tmp_x = random.randint(0, 9)
                tmp_y = random.randint(0, 9)
            else : 
                tmp_x, tmp_y = self.ai(2)
            self.tmp = self.tmp + 1
        else:
            tmp_x, tmp_y = self.ai(2)
            
        reward_tmp = self.put_stone(tmp_x, tmp_y, 2)
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
    
    def is_empty(self, x, y):
        return self.is_color(x, y, 0)

    def is_color(self, x, y, color):
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == color
    
    def grade(self, x, y, color):
        score = 0

        def matches(pattern, dir):
            p = pattern.index('_')
            for i in range(len(pattern)):
                xx = x + (i-p)*dir[0]
                yy = y + (i-p)*dir[1]
                if pattern[i] == ' ' and not self.is_empty(xx, yy):
                    return False
                if pattern[i] == 'O' and not self.is_color(xx, yy, color):
                    return False
                if pattern[i] == '|' and self.is_color(xx, yy, color):
                    return False
            return True

        matched = {}  # to avoid matching symmetrical patterns twice
        for dir in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
            for pattern, value in PATTERNS.items():
                if matched.get((-dir[0], -dir[1])) == pattern:
                    break
                if matches(pattern, dir):
                    score += value
                    matched[dir] = pattern
                    break

        if score >= 10**5 + 10**3:
            score += 10**7
        elif score >= 2 * 10**3:
            score += 10**6

        return score

    def ai(self, color):
        best, score = [], 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not self.is_empty(i, j):
                    continue
                v1 = self.grade(i, j, color)
                v2 = self.grade(i, j, 1 if color == 2 else 1)
                v = v1 + v2/10
                if v > score:
                    score = v
                    best = [(i, j)]
                elif v == score:
                    best.append((i, j))

        if best:
            pos = random.choice(best)
            x, y = pos
            return pos