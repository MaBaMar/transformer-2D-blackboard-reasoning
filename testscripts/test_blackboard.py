"""Script to play around with blackboard state generation"""

from projectlib.my_datasets._blackboard_operands import Subtraction
from projectlib.my_datasets.blackboards import BasicOpBlackboardIterator, BlackboardSpec, Addition

import numpy as np

def print_board(board):
    print("="*(len(board[0])+2))
    for l in board:
        print(f"|{l}|")
    print("="*(len(board[0])+2))

if __name__ == "__main__":

    x = np.array([0, 9, 9, 9])
    y = np.array([9, 9, 9, 9])

    spec = BlackboardSpec(15, 15, True, Addition())

    b = BasicOpBlackboardIterator(x, y, spec)
    for item in b:
        print_board(item)
