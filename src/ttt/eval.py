"""
Evaluation functions.

Tests model strength against random and minimax opponents,
and measures agreement with teacher on all legal states.
"""

import math
import random
from typing import Dict, List, Tuple

import torch

from .game import is_terminal, legal_moves, apply_move, side_to_move
from .minimax import minimax_value_and_moves, teacher_policy, iter_all_legal_nonterminal_states
from .model import board_to_tokens_perspective, legal_move_mask
from .train import masked_softmax


@torch.inference_mode()
def eval_vs_random(
    model: torch.nn.Module,
    device: torch.device,
    games: int = 500,
) -> Tuple[float, float, float]:
    """
    Evaluate model vs random opponent.

    Returns:
        (win_rate, draw_rate, loss_rate)
    """
    model.eval()
    wins = draws = losses = 0

    for g in range(games):
        board = [0] * 9
        player = +1
        model_side = +1 if (g % 2 == 0) else -1

        for _ in range(9):
            done, winner = is_terminal(board)
            if done:
                if winner == 0:
                    draws += 1
                elif winner == model_side:
                    wins += 1
                else:
                    losses += 1
                break

            if player == model_side:
                tokens = board_to_tokens_perspective(board, player).to(device).unsqueeze(0)
                mask = legal_move_mask(board).to(device).unsqueeze(0)
                logits, _ = model(tokens)
                logits = logits.squeeze(0)
                action = int(logits.masked_fill(~mask.squeeze(0), -1e9).argmax().item())
            else:
                action = random.choice(legal_moves(board))

            board[action] = player
            player = -player

    total = wins + draws + losses
    return wins / total, draws / total, losses / total


@torch.inference_mode()
def eval_vs_minimax(
    model: torch.nn.Module,
    device: torch.device,
    games: int = 500,
    teacher_plays_optimal_random: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model vs minimax teacher.

    Args:
        teacher_plays_optimal_random: If True, teacher randomly samples from optimal moves

    Returns:
        Dict with 'games', 'model_w', 'model_d', 'model_l'
    """
    model.eval()
    wins = draws = losses = 0

    for g in range(games):
        board = [0] * 9
        model_side = +1 if (g % 2 == 0) else -1

        while True:
            done, winner = is_terminal(board)
            if done:
                if winner == 0:
                    draws += 1
                elif winner == model_side:
                    wins += 1
                else:
                    losses += 1
                break

            stm = side_to_move(board)

            if stm == model_side:
                tokens = board_to_tokens_perspective(board, stm).to(device).unsqueeze(0)
                mask = legal_move_mask(board).to(device)
                logits, _ = model(tokens)
                logits = logits.squeeze(0)
                action = int(logits.masked_fill(~mask, -1e9).argmax().item())
            else:
                _, best_moves = minimax_value_and_moves(board, stm)
                if teacher_plays_optimal_random:
                    action = random.choice(best_moves)
                else:
                    action = best_moves[0]

            board[action] = stm

    total = wins + draws + losses
    return {
        "games": total,
        "model_w": wins / total,
        "model_d": draws / total,
        "model_l": losses / total,
    }


@torch.inference_mode()
def eval_teacher_agreement_all_states(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, object]:
    """
    Evaluate teacher agreement on all legal non-terminal states.

    This is exhaustive evaluation over the entire game tree.

    Returns:
        Dict with metrics and raw data (prefixed with '_')
    """
    model.eval()
    states = list(iter_all_legal_nonterminal_states())
    n = len(states)

    # Accumulators
    top1_opt = 0
    opt_mass_sum = 0.0
    ce_sum = 0.0
    kl_sum = 0.0
    v_mse_sum = 0.0
    v_mae_sum = 0.0
    v_exact = 0
    sign_ok = 0
    sign_n = 0

    val_errs = []
    opt_masses = []

    eps = 1e-12

    for i in range(0, n, batch_size):
        chunk = states[i:i + batch_size]

        # Batch prepare inputs
        tokens = torch.stack([board_to_tokens_perspective(b, p) for b, p in chunk], dim=0).to(device)
        masks = torch.stack([legal_move_mask(b) for b, _ in chunk], dim=0).to(device)

        # Teacher targets
        pi_star_list = []
        v_star_list = []
        for b, p in chunk:
            pi, v = teacher_policy(b, p)
            pi_star_list.append(pi)
            v_star_list.append(v)
        pi_star = torch.stack(pi_star_list, dim=0).to(device)
        v_star = torch.stack(v_star_list, dim=0).to(device)

        # Model predictions
        logits, v_pred = model(tokens)
        logp = logits  # Already have masking logic in model
        p = torch.softmax(logits, dim=-1)

        # Mask-aware softmax
        logp = logits.masked_fill(~masks, -1e9)
        logp = torch.log_softmax(logp, dim=-1)
        p = logp.exp()

        best = (pi_star > 0)
        a = p.argmax(dim=1)
        top1_opt += best.gather(1, a.unsqueeze(1)).squeeze(1).sum().item()

        opt_mass = (p * best.float()).sum(dim=1)
        opt_mass_sum += opt_mass.sum().item()

        ce = (-(pi_star * logp).sum(dim=1))
        ce_sum += ce.sum().item()

        log_pi = torch.log(torch.clamp(pi_star, min=eps))
        kl = (pi_star * (log_pi - logp)).sum(dim=1)
        kl_sum += kl.sum().item()

        # Value metrics
        diff = v_pred - v_star
        v_mse_sum += (diff * diff).sum().item()
        v_mae_sum += diff.abs().sum().item()
        v_round = torch.clamp(torch.round(v_pred), -1, 1)
        v_exact += (v_round == v_star).sum().item()

        non_draw = (v_star != 0)
        if non_draw.any():
            sign_ok += (torch.sign(v_pred[non_draw]) == torch.sign(v_star[non_draw])).sum().item()
            sign_n += non_draw.sum().item()

        val_errs.extend(diff.detach().cpu().tolist())
        opt_masses.extend(opt_mass.detach().cpu().tolist())

    # Distribution stats
    def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
    def std(xs):
        if not xs: return float("nan")
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    return {
        "teacher_n_states": n,
        "teacher_opt_top1_acc": top1_opt / n,
        "teacher_opt_mass_mean": opt_mass_sum / n,
        "teacher_policy_ce_mean": ce_sum / n,
        "teacher_kl_t_m_mean": kl_sum / n,
        "teacher_v_mse_mean": v_mse_sum / n,
        "teacher_v_mae_mean": v_mae_sum / n,
        "teacher_v_exact_acc": v_exact / n,
        "teacher_v_sign_acc_non_draw": (sign_ok / sign_n) if sign_n > 0 else float("nan"),
        "teacher_value_err_mean": mean(val_errs),
        "teacher_value_err_std": std(val_errs),
        "teacher_opt_mass_std": std(opt_masses),
        "_val_errs": val_errs,
        "_opt_masses": opt_masses,
    }


@torch.inference_mode()
def eval_vs_random_collect_losses(
    model: torch.nn.Module,
    device: torch.device,
    games: int,
    hard_buf,
) -> Tuple[float, float, float]:
    """
    Evaluate vs random and collect hard negatives from losses.

    Returns (win_rate, draw_rate, loss_rate).
    """
    model.eval()
    wins = draws = losses = 0

    for g in range(games):
        board = [0] * 9
        player = +1
        model_side = +1 if (g % 2 == 0) else -1
        model_positions = []

        for _ in range(9):
            done, winner = is_terminal(board)
            if done:
                if winner == 0:
                    draws += 1
                elif winner == model_side:
                    wins += 1
                else:
                    losses += 1

                # Collect hard negatives
                if hard_buf is not None and winner != 0 and winner != model_side:
                    for tok, m, pi, v in model_positions:
                        hard_buf.add(tok, m, pi, v)
                break

            lm = legal_moves(board)
            if player == model_side:
                tokens = board_to_tokens_perspective(board, player).to(device).unsqueeze(0)
                mask = legal_move_mask(board).to(device).unsqueeze(0)
                logits, _ = model(tokens)
                logits = logits.squeeze(0)
                mask1 = mask.squeeze(0)
                action = int(logits.masked_fill(~mask1, -1e9).argmax().item())

                pi_star, v_star = teacher_policy(board, player)
                model_positions.append((
                    tokens.squeeze(0).cpu(),
                    mask.squeeze(0).cpu(),
                    pi_star.cpu(),
                    v_star.cpu(),
                ))
            else:
                action = random.choice(lm)

            board[action] = player
            player = -player

    total = wins + draws + losses
    return wins / total, draws / total, losses / total
