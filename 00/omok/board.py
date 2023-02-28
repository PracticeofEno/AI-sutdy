
import random
from enum import Enum


# ' ' - empty field
# 'O' - our color
# '|' - anything except for our color
# '_' - position to grade
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


class Color(Enum):
    NONE = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self):
        return Color.WHITE if self == Color.BLACK else Color.BLACK


class Board:
    def __init__(self, size):
        self.size = size
        self.fields = [[Color.NONE for i in range(size)] for j in range(size)]

    def is_empty(self, x, y):
        return self.is_color(x, y, Color.NONE)

    def is_color(self, x, y, color):
        return 0 <= x < self.size and 0 <= y < self.size and self.fields[x][y] == color

    def put(self, x, y, color):
        self.fields[x][y] = color

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

        for i in range(self.size):
            for j in range(self.size):
                if not self.is_empty(i, j):
                    continue
                v1 = self.grade(i, j, color)
                v2 = self.grade(i, j, color.opponent)
                v = v1 + v2/10
                if v > score:
                    score = v
                    best = [(i, j)]
                elif v == score:
                    best.append((i, j))

        if best:
            pos = random.choice(best)
            x, y = pos
            self.put(x, y, color)
            return pos

    @property
    def winner(self):
        for i in range(self.size):
            for j in range(self.size):
                for dir in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
                    try:
                        s = set(self.fields[i+k*dir[0]][j+k*dir[1]] for k in range(5))
                    except IndexError:
                        continue
                    if len(s) > 1:
                        continue
                    color = s.pop()
                    if color != Color.NONE and not self.is_color(i-dir[0], j-dir[1], color) and not self.is_color(i+5*dir[0], j+5*dir[1], color):
                        return color
        return Color.NONE
