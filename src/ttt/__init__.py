"""
Tiny Transformer TicTacToe - Train a transformer to play perfect TicTacToe.

This package implements AlphaZero-style training with exact minimax supervision
(no MCTS required for this simple game).
"""

from .model import TinyTTTTransformer, board_to_tokens_perspective, legal_move_mask
from .game import GameState, is_terminal, legal_moves, apply_move, side_to_move
from .minimax import minimax_value_and_moves, teacher_policy, iter_all_legal_nonterminal_states
from .symmetries import apply_symmetry_board, apply_symmetry_policy, SYM_MAPS
from .replay import PositionReplay, HardNegativeReplay
from .train import (
    TrainConfig,
    compute_loss,
    compute_metrics,
    play_self_play_game_collect_positions,
    param_norm,
    get_lr,
)
from .eval import (
    eval_vs_random,
    eval_vs_minimax,
    eval_teacher_agreement_all_states,
    eval_vs_random_collect_losses,
)

__version__ = "0.1.0"
__all__ = [
    "TinyTTTTransformer",
    "board_to_tokens_perspective",
    "legal_move_mask",
    "GameState",
    "is_terminal",
    "legal_moves",
    "apply_move",
    "side_to_move",
    "minimax_value_and_moves",
    "teacher_policy",
    "iter_all_legal_nonterminal_states",
    "TrainConfig",
    "compute_loss",
    "compute_metrics",
    "play_self_play_game_collect_positions",
    "param_norm",
    "get_lr",
    "eval_vs_random",
    "eval_vs_minimax",
    "eval_teacher_agreement_all_states",
    "eval_vs_random_collect_losses",
]
