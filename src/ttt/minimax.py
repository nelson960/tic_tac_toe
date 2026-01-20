"""
Exact minimax solver for TicTacToe with caching.

Provides provably optimal policy and value targets for training.
"""

import torch
from typing import Dict, List, Tuple

from .game import is_terminal, legal_moves, apply_move


# Cache: (board_tuple, player) -> (value, best_moves_tuple)
_MINIMAX_CACHE: Dict[Tuple[Tuple[int, ...], int], Tuple[int, Tuple[int, ...]]] = {}


def minimax_value_and_moves(board: List[int], player: int) -> Tuple[int, List[int]]:
    """
    Compute minimax value and best moves from current state.

    Args:
        board: Current board state
        player: Current player (+1 or -1)

    Returns:
        (value, best_moves) where:
        - value: +1 (win), 0 (draw), -1 (loss) from current player's perspective
        - best_moves: list of actions achieving optimal value
    """
    key = (tuple(board), player)
    if key in _MINIMAX_CACHE:
        v, best = _MINIMAX_CACHE[key]
        return v, list(best)

    done, winner = is_terminal(board)
    if done:
        if winner == 0:
            v = 0
        elif winner == player:
            v = +1
        else:
            v = -1
        _MINIMAX_CACHE[key] = (v, tuple())
        return v, []

    best_v = -2
    best_moves: List[int] = []

    for action in legal_moves(board):
        next_board = apply_move(board, player, action)
        child_v, _ = minimax_value_and_moves(next_board, -player)
        v_here = -child_v  # Negate for opponent's perspective

        if v_here > best_v:
            best_v = v_here
            best_moves = [action]
        elif v_here == best_v:
            best_moves.append(action)

    _MINIMAX_CACHE[key] = (best_v, tuple(best_moves))
    return best_v, best_moves


def teacher_policy(board: List[int], player: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal teacher targets (policy and value).

    Returns:
        pi_star: [9] tensor with uniform distribution over optimal moves
        v_star: scalar tensor in {-1, 0, +1}
    """
    v, best_moves = minimax_value_and_moves(board, player)

    pi = torch.zeros(9, dtype=torch.float32)
    if best_moves:
        pi[best_moves] = 1.0 / len(best_moves)

    return pi, torch.tensor(float(v), dtype=torch.float32)


def clear_cache():
    """Clear minimax cache (useful for memory management)."""
    global _MINIMAX_CACHE
    _MINIMAX_CACHE.clear()


def cache_size() -> int:
    """Return current cache size."""
    return len(_MINIMAX_CACHE)


def iter_all_legal_nonterminal_states():
    """
    Iterate over all legal non-terminal board states.

    Yields:
        (board, player) tuples for exhaustive evaluation.
    """
    for n in range(3**9):
        # Decode base-3 representation
        x = n
        digits = [0] * 9
        for i in range(9):
            digits[i] = x % 3
            x //= 3

        x_cnt = sum(1 for d in digits if d == 1)
        o_cnt = sum(1 for d in digits if d == 2)

        # Legal turn order: X starts
        if not (x_cnt == o_cnt or x_cnt == o_cnt + 1):
            continue

        # Build board
        board = [0] * 9
        for i, d in enumerate(digits):
            if d == 1:
                board[i] = +1
            elif d == 2:
                board[i] = -1

        # Skip illegal states
        from .game import winners_set, is_terminal
        if len(winners_set(board)) >= 2:
            continue

        # Skip terminal
        done, _ = is_terminal(board)
        if done:
            continue

        # Side to move
        player = +1 if x_cnt == o_cnt else -1
        yield board, player
