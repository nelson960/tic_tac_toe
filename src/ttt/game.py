"""
TicTacToe game rules and state management.

Board representation: list[int] of length 9
  - 0: empty
  - +1: X
  - -1: O

Player: +1 (X) or -1 (O) - side to move
"""

from typing import List, Tuple
from dataclasses import dataclass

# Winning lines (rows, columns, diagonals)
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


@dataclass
class GameState:
    """Immutable game state."""
    board: List[int]
    player: int  # Side to move: +1 or -1


def winners_set(board: List[int]) -> set:
    """Return set of winners (+1, -1, or both if illegal)."""
    wins = set()
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            wins.add(+1)
        elif s == -3:
            wins.add(-1)
    return wins


def is_terminal(board: List[int]) -> Tuple[bool, int]:
    """
    Check if board is terminal.

    Returns:
        (is_terminal, winner) where winner is +1/-1/0
    """
    wset = winners_set(board)
    if len(wset) >= 2:
        # Illegal board state (both win) - treat as draw
        return True, 0
    if len(wset) == 1:
        return True, next(iter(wset))
    if all(v != 0 for v in board):
        return True, 0
    return False, 0


def legal_moves(board: List[int]) -> List[int]:
    """Return list of legal move indices (empty squares)."""
    return [i for i, v in enumerate(board) if v == 0]


def apply_move(board: List[int], player: int, action: int) -> List[int]:
    """Apply move and return new board."""
    new_board = board[:]
    new_board[action] = player
    return new_board


def side_to_move(board: List[int]) -> int:
    """Infer side to move from board state (X plays first)."""
    x_cnt = sum(1 for v in board if v == +1)
    o_cnt = sum(1 for v in board if v == -1)
    return +1 if x_cnt == o_cnt else -1


def is_legal_board(board: List[int]) -> bool:
    """Check if board respects game rules."""
    x_cnt = sum(1 for v in board if v == +1)
    o_cnt = sum(1 for v in board if v == -1)

    # X goes first, so x_cnt == o_cnt or x_cnt == o_cnt + 1
    if not (x_cnt == o_cnt or x_cnt == o_cnt + 1):
        return False

    # Can't have both winners
    wset = winners_set(board)
    if len(wset) >= 2:
        return False

    return True
