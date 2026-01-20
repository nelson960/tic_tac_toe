"""
D4 symmetry augmentation for TicTacToe (8 transforms).

Rotations: 0°, 90°, 180°, 270°
Reflections: horizontal, vertical, main diagonal, anti-diagonal
"""

import torch
from typing import List


def _idx(r: int, c: int) -> int:
    """Convert (row, col) to flat index."""
    return r * 3 + c


def _build_symmetry_maps():
    """Build 8 permutation maps for D4 symmetries."""
    maps = []
    for k in range(8):
        mp = [0] * 9
        for r in range(3):
            for c in range(3):
                # Apply transform k
                if k == 0:   rt, ct = r, c                # identity
                elif k == 1: rt, ct = c, 2 - r            # rotate 90
                elif k == 2: rt, ct = 2 - r, 2 - c        # rotate 180
                elif k == 3: rt, ct = 2 - c, r            # rotate 270
                elif k == 4: rt, ct = r, 2 - c            # reflect horizontal
                elif k == 5: rt, ct = 2 - r, c            # reflect vertical
                elif k == 6: rt, ct = c, r                # reflect main diag
                else:        rt, ct = 2 - c, 2 - r        # reflect anti-diag
                mp[_idx(rt, ct)] = _idx(r, c)
        maps.append(torch.tensor(mp, dtype=torch.long))
    return maps


# Pre-computed symmetry maps
SYM_MAPS = _build_symmetry_maps()


def apply_symmetry_board(board: List[int], sym_id: int) -> List[int]:
    """
    Apply symmetry transform to board.

    Args:
        board: [9] list of board values
        sym_id: symmetry ID (0-7)

    Returns:
        Transformed board
    """
    mp = SYM_MAPS[sym_id]
    return [board[int(mp[i].item())] for i in range(9)]


def apply_symmetry_policy(pi: torch.Tensor, sym_id: int) -> torch.Tensor:
    """
    Apply symmetry transform to policy distribution.

    Args:
        pi: [9] policy tensor
        sym_id: symmetry ID (0-7)

    Returns:
        Transformed policy
    """
    mp = SYM_MAPS[sym_id].to(pi.device)
    return pi.index_select(0, mp)


def apply_symmetry_mask(mask: torch.Tensor, sym_id: int) -> torch.Tensor:
    """
    Apply symmetry transform to legal move mask.

    Args:
        mask: [9] boolean mask
        sym_id: symmetry ID (0-7)

    Returns:
        Transformed mask
    """
    mp = SYM_MAPS[sym_id].to(mask.device)
    return mask.index_select(0, mp)


def get_all_symmetries(board: List[int]) -> List[List[int]]:
    """Return all 8 symmetric versions of a board."""
    return [apply_symmetry_board(board, k) for k in range(8)]


def get_random_symmetry(board: List[int], rng) -> List[int]:
    """Return a random symmetric version of the board."""
    sym_id = rng.integers(0, 8)
    return apply_symmetry_board(board, sym_id)
