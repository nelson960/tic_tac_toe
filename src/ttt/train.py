"""
Training utilities: loss functions, metrics, self-play.
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from .game import is_terminal, legal_moves, apply_move, side_to_move
from .minimax import teacher_policy
from .symmetries import apply_symmetry_board, apply_symmetry_policy, apply_symmetry_mask
from .replay import PositionReplay


@dataclass
class TrainConfig:
    """Training configuration."""

    # Random seed
    seed: int = 0

    # Training steps
    steps: int = 3000

    # Self-play per step
    games_per_step: int = 64

    # Temperature annealing
    temp_start: float = 1.3
    temp_end: float = 0.6

    # Dirichlet exploration
    dir_alpha: float = 0.3
    dir_eps: float = 0.25

    # Replay buffers
    replay_max_positions: int = 200_000
    hard_max_positions: int = 50_000
    hard_frac: float = 0.4

    # Training batch
    batch_size: int = 2048

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss weights
    value_coef: float = 1.0
    beta_entropy: float = 0.01

    # Augmentation
    use_symmetry: bool = True

    # Logging
    print_every: int = 100
    eval_every: int = 300

    # Paths
    save_dir: str = "runs"


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax with legal move masking."""
    masked = logits.masked_fill(~legal_mask, -1e9)
    return F.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Softmax with legal move masking."""
    masked = logits.masked_fill(~legal_mask, -1e9)
    return F.softmax(masked, dim=-1)


def compute_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    legal_mask: torch.Tensor,
    pi_star: torch.Tensor,
    v_star: torch.Tensor,
    value_coef: float = 1.0,
    beta_entropy: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined loss.

    Returns:
        (total_loss, metrics_dict)
    """
    logp = masked_log_softmax(logits, legal_mask)
    probs = logp.exp()

    # Policy loss: CE with teacher
    policy_loss = -(pi_star * logp).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(values, v_star)

    # Entropy regularization
    entropy = -(probs * logp).sum(dim=1).mean()

    # Total
    total = policy_loss + value_coef * value_loss - beta_entropy * entropy

    return total, {
        "loss": float(total.item()),
        "pi_loss": float(policy_loss.item()),
        "v_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
    }


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    values: torch.Tensor,
    legal_mask: torch.Tensor,
    pi_star: torch.Tensor,
    v_star: torch.Tensor,
) -> Dict[str, float]:
    """Compute training metrics."""
    logp = masked_log_softmax(logits, legal_mask)
    p = logp.exp()

    best = (pi_star > 0)
    a = p.argmax(dim=1)

    # Top-1 optimal accuracy
    top1_opt = best.gather(1, a.unsqueeze(1)).squeeze(1).float().mean().item()

    # Optimal mass
    opt_mass = (p * best.float()).sum(dim=1).mean().item()

    # Cross-entropy
    ce = (-(pi_star * logp).sum(dim=1)).mean().item()

    # KL
    eps = 1e-12
    log_pi = torch.log(torch.clamp(pi_star, min=eps))
    kl = (pi_star * (log_pi - logp)).sum(dim=1).mean().item()

    # Value metrics
    v_mse = F.mse_loss(values, v_star).item()
    v_mae = F.l1_loss(values, v_star).item()
    v_round = torch.clamp(torch.round(values), -1, 1)
    v_exact = (v_round == v_star).float().mean().item()

    non_draw = (v_star != 0)
    if non_draw.any():
        v_sign_acc = (torch.sign(values[non_draw]) == torch.sign(v_star[non_draw])).float().mean().item()
    else:
        v_sign_acc = float("nan")

    return {
        "train_top1_opt": top1_opt,
        "train_opt_mass": opt_mass,
        "train_ce": ce,
        "train_kl_t_m": kl,
        "train_v_mse": v_mse,
        "train_v_mae": v_mae,
        "train_v_exact": v_exact,
        "train_v_sign_acc": v_sign_acc,
        "train_pmax_mean": p.max(dim=1).values.mean().item(),
    }


@torch.no_grad()
def play_self_play_game_collect_positions(
    model: nn.Module,
    device: torch.device,
    temp: float,
    dir_alpha: float,
    dir_eps: float,
    augment_sym: bool,
    replay: PositionReplay,
) -> Tuple[int, int, int]:
    """
    Play one self-play game and collect training positions.

    Returns:
        (winner, num_moves, positions_added)
    """
    before = len(replay)
    board = [0] * 9
    player = +1

    for ply in range(9):
        done, winner = is_terminal(board)
        if done:
            return winner, ply, len(replay) - before

        # Get board perspective tokens
        from .model import board_to_tokens_perspective, legal_move_mask
        tokens = board_to_tokens_perspective(board, player).to(device).unsqueeze(0)
        mask = legal_move_mask(board).to(device).unsqueeze(0)

        # Forward
        logits, _ = model(tokens)
        logits = logits.squeeze(0)
        mask = mask.squeeze(0)

        # Sample action
        probs = masked_softmax(logits / max(temp, 1e-6), mask)

        if dir_eps > 0:
            legal_idx = torch.where(mask)[0]
            if legal_idx.numel() > 0:
                # Sample Dirichlet noise (use CPU for unsupported devices)
                noise_device = device
                if device.type in ('mps', 'meta'):  # MPS and similar don't support Dirichlet
                    noise_device = torch.device('cpu')
                noise = torch.distributions.Dirichlet(
                    torch.full((legal_idx.numel(),), dir_alpha, device=noise_device)
                ).sample()
                if noise_device != device:
                    noise = noise.to(device)
                probs2 = probs.clone()
                probs2[legal_idx] = (1 - dir_eps) * probs[legal_idx] + dir_eps * noise
                probs = probs2 / probs2.sum()

        action = int(torch.multinomial(probs, 1).item())

        # Store with teacher targets
        if augment_sym:
            for sym_id in range(8):
                b_sym = apply_symmetry_board(board, sym_id)
                tok_sym = board_to_tokens_perspective(b_sym, player)
                m_sym = legal_move_mask(b_sym)
                pi_sym, v_sym = teacher_policy(b_sym, player)
                replay.add(tok_sym, m_sym, pi_sym, v_sym)
        else:
            pi_star, v_star = teacher_policy(board, player)
            replay.add(tokens.squeeze(0).cpu(), mask.squeeze(0).cpu(), pi_star.cpu(), v_star.cpu())

        # Apply move
        board[action] = player
        player = -player

    done, winner = is_terminal(board)
    return winner, 9, len(replay) - before


def param_norm(model: nn.Module) -> float:
    """Compute L2 norm of all parameters."""
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p is not None:
                total += float((p.detach() ** 2).sum().item())
    return total ** 0.5


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate."""
    for g in optimizer.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0
