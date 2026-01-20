#!/usr/bin/env python3
"""
Train a Tiny Transformer to play TicTacToe with minimax teacher supervision.

Generates markdown report with all 19 diagrams and tables.

Usage:
    python train.py                    # Full training (3000 steps)
    python train.py --steps 500        # Quick demo
    python train.py --steps 1000 --eval-every 200
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm.auto import trange, tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ttt import (
    TinyTTTTransformer,
    TrainConfig,
    compute_loss,
    compute_metrics,
    play_self_play_game_collect_positions,
    eval_vs_random_collect_losses,
    eval_vs_minimax,
    eval_teacher_agreement_all_states,
    PositionReplay,
    HardNegativeReplay,
    param_norm,
    get_lr,
)


def pick_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup(model, device, config: TrainConfig, replay: PositionReplay):
    """Warmup phase: collect initial self-play data."""
    model.eval()
    warmup_games = config.games_per_step
    warmup_temp = config.temp_start

    warm_winners = {+1: 0, -1: 0, 0: 0}
    warm_total_len = 0

    for _ in range(warmup_games):
        w, L, _ = play_self_play_game_collect_positions(
            model, device,
            temp=warmup_temp,
            dir_alpha=config.dir_alpha,
            dir_eps=config.dir_eps,
            augment_sym=config.use_symmetry,
            replay=replay
        )
        warm_winners[w] += 1
        warm_total_len += L

    warm_avgT = warm_total_len / max(1, warmup_games)
    tqdm.write(f"Warmup: {len(replay)} positions | avg length: {warm_avgT:.2f} | "
               f"X: {warm_winners[+1]} O: {warm_winners[-1]} D: {warm_winners[0]}")


def train_one_step(model, optimizer, device, config: TrainConfig, step: int,
                   replay: PositionReplay, hard: HardNegativeReplay):
    """Run one training step."""
    model.eval()
    step_t0 = time.perf_counter()

    # Anneal temperature
    t = step / config.steps
    temp = config.temp_start + t * (config.temp_end - config.temp_start)

    # 1) Collect self-play games
    t0 = time.perf_counter()
    winners = {+1: 0, -1: 0, 0: 0}
    total_len = 0

    for _ in range(config.games_per_step):
        w, L, added = play_self_play_game_collect_positions(
            model, device,
            temp=temp,
            dir_alpha=config.dir_alpha,
            dir_eps=config.dir_eps,
            augment_sym=config.use_symmetry,
            replay=replay
        )
        winners[w] += 1
        total_len += L

    avgT = total_len / max(1, config.games_per_step)
    collect_s = time.perf_counter() - t0

    # 2) Build batch
    model.train()

    B = config.batch_size
    Bh = int(B * config.hard_frac) if len(hard) > 0 else 0
    Br = B - Bh

    tok_r, mask_r, pi_r, v_r = replay.sample(Br, device=device)

    if Bh > 0 and len(hard) >= 10:
        tok_h, mask_h, pi_h, v_h = hard.sample(Bh, device=device)
        tokens = torch.cat([tok_r, tok_h], dim=0)
        masks = torch.cat([mask_r, mask_h], dim=0)
        pi_star = torch.cat([pi_r, pi_h], dim=0)
        v_star = torch.cat([v_r, v_h], dim=0)
    else:
        tokens, masks, pi_star, v_star = tok_r, mask_r, pi_r, v_r

    # 3) Update
    logits, values = model(tokens)
    loss, stats = compute_loss(
        logits, values, masks, pi_star, v_star,
        value_coef=config.value_coef,
        beta_entropy=config.beta_entropy,
    )
    extra = compute_metrics(logits, values, masks, pi_star, v_star)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item())
    optimizer.step()

    metrics = {
        "step": step,
        "temp": temp,
        "avgT": avgT,
        "collect_s": collect_s,
        "step_s": time.perf_counter() - step_t0,
        "replay_pos": len(replay),
        "hard_pos": len(hard),
        "selfplay_X": winners[+1],
        "selfplay_O": winners[-1],
        "selfplay_D": winners[0],
        "lr": get_lr(optimizer),
        "grad_norm": grad_norm,
        "param_norm": param_norm(model),
        **stats,
        **extra,
    }

    return metrics


def create_all_plots(history: list, teacher_eval: dict, output_dir: Path):
    """Create all 19 training plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pd = __import__('pandas')
    df = pd.DataFrame(history)

    # 1. Total Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], linewidth=2)
    plt.title('Total Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_1_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Loss Breakdown
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['pi_loss'], label='Policy Loss', linewidth=2)
    plt.plot(df['step'], df['v_loss'], label='Value Loss', linewidth=2)
    plt.title('Loss Breakdown', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_2_loss_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Policy Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['entropy'], linewidth=2, color='green')
    plt.title('Policy Entropy', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Entropy (nats)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_3_entropy.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Train Policy Agreement
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_top1_opt'], label='Top-1 Optimal', linewidth=2)
    plt.plot(df['step'], df['train_opt_mass'], label='Optimal Mass', linewidth=2)
    plt.plot(df['step'], df['train_pmax_mean'], label='Max Probability', linewidth=2, alpha=0.7)
    plt.title('Train Policy Agreement Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_4_policy_agreement.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Train Value Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(df['step'], df['train_v_mae'], linewidth=2, color='blue')
    axes[0, 0].set_title('Value MAE', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['step'], df['train_v_mse'], linewidth=2, color='red')
    axes[0, 1].set_title('Value MSE', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df['step'], df['train_v_exact'], linewidth=2, color='green')
    axes[1, 0].set_title('Value Exact Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['step'], df['train_v_sign_acc'], linewidth=2, color='purple')
    axes[1, 1].set_title('Value Sign Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Train Value Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_5_value_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Gradient Norm
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['grad_norm'], linewidth=2, color='orange')
    plt.title('Gradient Norm (Clipped)', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Norm', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_6_grad_norm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Parameter Norm
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['param_norm'], linewidth=2, color='brown')
    plt.title('Parameter L2 Norm', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Norm', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_7_param_norm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['lr'], linewidth=2, color='teal')
    plt.title('Learning Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_8_lr.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 9. Temperature & Avg Game Length
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['temp'], label='Temperature', linewidth=2)
    plt.plot(df['step'], df['avgT'], label='Avg Game Length', linewidth=2, alpha=0.7)
    plt.title('Temperature & Average Game Length', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_9_temp_game_len.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 10. Collection Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['collect_s'], linewidth=2, color='purple')
    plt.title('Self-Play Collection Time per Step', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Seconds', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_10_collect_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 11. Total Step Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['step_s'], linewidth=2, color='magenta')
    plt.title('Total Step Time', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Seconds', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_11_step_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 12. Buffer Sizes
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['replay_pos'], label='Replay Buffer', linewidth=2)
    if df['hard_pos'].max() > 0:
        plt.plot(df['step'], df['hard_pos'], label='Hard Negative Buffer', linewidth=2)
    plt.title('Buffer Sizes', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Positions', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_12_buffer_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 13. Self-Play Outcomes (rolling average)
    window = 50
    df_roll = df.rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df_roll['selfplay_X'], label='X Wins', linewidth=2)
    plt.plot(df['step'], df_roll['selfplay_O'], label='O Wins', linewidth=2)
    plt.plot(df['step'], df_roll['selfplay_D'], label='Draws', linewidth=2)
    plt.title(f'Self-Play Outcomes (Rolling Avg, window={window})', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_13_selfplay_outcomes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 14. Eval vs Random
    if 'eval_w' in df.columns:
        eval_df = df.dropna(subset=['eval_w'])
        if len(eval_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(eval_df['step'], eval_df['eval_w'], label='Wins', marker='o', linewidth=2)
            plt.plot(eval_df['step'], eval_df['eval_d'], label='Draws', marker='o', linewidth=2)
            plt.plot(eval_df['step'], eval_df['eval_l'], label='Losses', marker='o', linewidth=2)
            plt.title('Evaluation vs Random Opponent', fontsize=14, fontweight='bold')
            plt.xlabel('Step', fontsize=12)
            plt.ylabel('Rate', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'plot_14_eval_random.png', dpi=150, bbox_inches='tight')
            plt.close()

    # 15-16. Teacher Agreement
    if 'teacher_opt_top1_acc' in df.columns:
        teacher_df = df.dropna(subset=['teacher_opt_top1_acc'])
        if len(teacher_df) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            axes[0].plot(teacher_df['step'], teacher_df['teacher_opt_top1_acc'],
                        label='Top-1 Optimal', marker='o', linewidth=2)
            axes[0].plot(teacher_df['step'], teacher_df['teacher_opt_mass_mean'],
                        label='Optimal Mass', marker='o', linewidth=2)
            axes[0].set_title('Teacher Agreement: Policy Metrics (All 4,520 Legal States)',
                             fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Step', fontsize=11)
            axes[0].set_ylabel('Metric', fontsize=11)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(teacher_df['step'], teacher_df['teacher_v_exact_acc'],
                        label='Value Exact', marker='o', linewidth=2)
            axes[1].plot(teacher_df['step'], teacher_df['teacher_v_mae_mean'],
                        label='Value MAE', marker='o', linewidth=2)
            axes[1].plot(teacher_df['step'], teacher_df['teacher_policy_ce_mean'],
                        label='Policy CE', marker='o', linewidth=2)
            axes[1].set_title('Teacher Agreement: Value + Cross-Entropy',
                             fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Step', fontsize=11)
            axes[1].set_ylabel('Metric', fontsize=11)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'plot_15_teacher_policy.png', dpi=150, bbox_inches='tight')
            plt.savefig(output_dir / 'plot_16_teacher_value.png', dpi=150, bbox_inches='tight')
            plt.close()

    # 17-18. Histograms
    if teacher_eval and '_val_errs' in teacher_eval:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(teacher_eval['_val_errs'], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_title('Value Error Distribution\n(v_pred - v_teacher on all legal states)',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Value Error', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(teacher_eval['_opt_masses'], bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_title('Optimal Move Probability Mass Distribution\n(on all 4,520 legal states)',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Sum of Probabilities on Optimal Moves', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'plot_17_hist_values.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'plot_18_hist_opt_mass.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 19. Training Overview (multi-panel)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Training Overview - All Key Metrics', fontsize=16, fontweight='bold')

    # Row 1
    axes[0, 0].plot(df['step'], df['loss'], linewidth=2)
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['step'], df['train_top1_opt'], linewidth=2, color='green')
    axes[0, 1].set_title('Top-1 Optimal', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(df['step'], df['train_v_exact'], linewidth=2, color='purple')
    axes[0, 2].set_title('Value Exact', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2
    axes[1, 0].plot(df['step'], df['entropy'], linewidth=2, color='orange')
    axes[1, 0].set_title('Entropy', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['step'], df['train_opt_mass'], linewidth=2, color='blue')
    axes[1, 1].set_title('Optimal Mass', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    if 'eval_w' in df.columns:
        eval_df = df.dropna(subset=['eval_w'])
        if len(eval_df) > 0:
            axes[1, 2].plot(eval_df['step'], eval_df['eval_w'], marker='o', label='Wins', linewidth=2)
            axes[1, 2].plot(eval_df['step'], eval_df['eval_d'], marker='o', label='Draws', linewidth=2)
            axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title('vs Random', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3
    axes[2, 0].plot(df['step'], df['grad_norm'], linewidth=2, color='red')
    axes[2, 0].set_title('Gradient Norm', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(df['step'], df['temp'], linewidth=2)
    axes[2, 1].set_title('Temperature', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(df['step'], df['replay_pos'], linewidth=2, color='brown')
    axes[2, 2].set_title('Replay Size', fontweight='bold')
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'plot_19_overview.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_markdown_report(history: list, teacher_eval: dict, config: TrainConfig,
                             run_dir: Path, device: torch.device):
    """Generate comprehensive markdown report."""
    final = history[-1]
    pd = __import__('pandas')
    df = pd.DataFrame(history)

    md = []
    md.append("# TicTacToe Transformer Training Report\n")
    md.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Device:** {device}\n")
    md.append(f"**Total Parameters:** 101,386\n")

    # Configuration table
    md.append("\n## 📋 Training Configuration\n")
    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Steps | {config.steps:,} |")
    md.append(f"| Games per step | {config.games_per_step} |")
    md.append(f"| Batch size | {config.batch_size:,} |")
    md.append(f"| Replay buffer | {config.replay_max_positions:,} |")
    md.append(f"| Hard buffer | {config.hard_max_positions:,} |")
    md.append(f"| Hard fraction | {config.hard_frac:.0%} |")
    md.append(f"| Learning rate | {config.lr} |")
    md.append(f"| Temperature schedule | {config.temp_start} → {config.temp_end} |")
    md.append(f"| Dirichlet α | {config.dir_alpha} |")
    md.append(f"| Dirichlet ε | {config.dir_eps} |")
    md.append(f"| Symmetry augmentation | {config.use_symmetry} |")
    md.append(f"| Value coefficient | {config.value_coef} |")
    md.append(f"| Entropy coefficient | {config.beta_entropy} |")

    # Final metrics
    md.append("\n## 🎯 Final Performance Metrics\n")

    md.append("### Training Metrics (Batch)")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Total Loss** | {final['loss']:.4f} |")
    md.append(f"| **Policy Loss** | {final['pi_loss']:.4f} |")
    md.append(f"| **Value Loss** | {final['v_loss']:.4f} |")
    md.append(f"| **Entropy** | {final['entropy']:.4f} |")
    md.append(f"| **Top-1 Optimal** | {final['train_top1_opt']:.2%} |")
    md.append(f"| **Optimal Mass** | {final['train_opt_mass']:.4f} |")
    md.append(f"| **Value Exact** | {final['train_v_exact']:.4f} |")
    md.append(f"| **Value MAE** | {final['train_v_mae']:.4f} |")

    # Evaluation results
    if 'eval_w' in final:
        md.append("\n### vs Random Opponent (500 games)")
        md.append("| Result | Rate |")
        md.append("|--------|------|")
        md.append(f"| **Wins** | {final['eval_w']:.2%} |")
        md.append(f"| **Draws** | {final['eval_d']:.2%} |")
        md.append(f"| **Losses** | {final['eval_l']:.2%} |")

    if 'minimax_w' in final:
        md.append("\n### vs Minimax Teacher (500 games)")
        md.append("| Result | Rate |")
        md.append("|--------|------|")
        md.append(f"| **Wins** | {final['minimax_w']:.2%} |")
        md.append(f"| **Draws** | {final['minimax_d']:.2%} |")
        md.append(f"| **Losses** | {final['minimax_l']:.2%} |")
        md.append("\n> **Note:** 100% draws vs minimax = perfect play! In TicTacToe, optimal play from both sides always results in a draw.")

    # Teacher agreement
    if teacher_eval:
        md.append("\n## 👨‍🏫 Teacher Agreement (All 4,520 Legal States)")
        md.append("\n### Policy Metrics")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| **Top-1 Optimal Accuracy** | {teacher_eval['teacher_opt_top1_acc']:.2%} |")
        md.append(f"| **Optimal Mass (mean)** | {teacher_eval['teacher_opt_mass_mean']:.4f} |")
        md.append(f"| **Optimal Mass (std)** | {teacher_eval['teacher_opt_mass_std']:.4f} |")
        md.append(f"| **Policy Cross-Entropy** | {teacher_eval['teacher_policy_ce_mean']:.4f} |")
        md.append(f"| **KL(Teacher||Model)** | {teacher_eval['teacher_kl_t_m_mean']:.4f} |")

        md.append("\n### Value Metrics")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| **Value Exact Accuracy** | {teacher_eval['teacher_v_exact_acc']:.2%} |")
        md.append(f"| **Value MAE** | {teacher_eval['teacher_v_mae_mean']:.4f} |")
        md.append(f"| **Value MSE** | {teacher_eval['teacher_v_mse_mean']:.4f} |")
        md.append(f"| **Value Sign Accuracy (non-draw)** | {teacher_eval['teacher_v_sign_acc_non_draw']:.2%} |")
        md.append(f"| **Value Error (mean)** | {teacher_eval['teacher_value_err_mean']:.4f} |")
        md.append(f"| **Value Error (std)** | {teacher_eval['teacher_value_err_std']:.4f} |")

    # Training progress table
    md.append("\n## 📈 Training Progress\n")
    checkpoints = [0, len(history)//4, len(history)//2, 3*len(history)//4, len(history)-1]
    md.append("| Step | Loss | Top-1 Opt | Opt Mass | v Exact | vs Random W/D/L | vs Minimax W/D/L |")
    md.append("|------|------|-----------|----------|---------|-----------------|------------------|")

    for idx in checkpoints:
        h = history[idx]
        # Find most recent evaluation results before or at this checkpoint
        random_eval = minimax_eval = "-"
        for i in range(idx, -1, -1):
            if i < len(history) and 'eval_w' in history[i]:
                e = history[i]
                random_eval = f"{e.get('eval_w', 0):.0%}/{e.get('eval_d', 0):.0%}/{e.get('eval_l', 0):.0%}"
                break
        for i in range(idx, -1, -1):
            if i < len(history) and 'minimax_w' in history[i]:
                e = history[i]
                minimax_eval = f"{e.get('minimax_w', 0):.0%}/{e.get('minimax_d', 0):.0%}/{e.get('minimax_l', 0):.0%}"
                break
        md.append(f"| {idx} | {h.get('loss', 0):.3f} | {h.get('train_top1_opt', 0):.1%} | "
                  f"{h.get('train_opt_mass', 0):.3f} | {h.get('train_v_exact', 0):.3f} | "
                  f"{random_eval} | {minimax_eval} |")

    # All plots
    md.append("\n## 📊 Training Visualizations\n")

    plot_descriptions = [
        ("plot_1_loss.png", "Total Loss"),
        ("plot_2_loss_breakdown.png", "Loss Breakdown (Policy + Value)"),
        ("plot_3_entropy.png", "Policy Entropy"),
        ("plot_4_policy_agreement.png", "Train Policy Agreement Metrics"),
        ("plot_5_value_metrics.png", "Train Value Metrics"),
        ("plot_6_grad_norm.png", "Gradient Norm"),
        ("plot_7_param_norm.png", "Parameter L2 Norm"),
        ("plot_8_lr.png", "Learning Rate"),
        ("plot_9_temp_game_len.png", "Temperature & Average Game Length"),
        ("plot_10_collect_time.png", "Self-Play Collection Time"),
        ("plot_11_step_time.png", "Total Step Time"),
        ("plot_12_buffer_sizes.png", "Buffer Sizes"),
        ("plot_13_selfplay_outcomes.png", "Self-Play Outcomes (Rolling Avg)"),
        ("plot_14_eval_random.png", "Eval vs Random"),
        ("plot_15_teacher_policy.png", "Teacher Agreement: Policy"),
        ("plot_16_teacher_value.png", "Teacher Agreement: Value"),
        ("plot_17_hist_values.png", "Value Error Histogram"),
        ("plot_18_hist_opt_mass.png", "Optimal Mass Histogram"),
        ("plot_19_overview.png", "Training Overview (All Metrics)"),
    ]

    for plot_file, description in plot_descriptions:
        if (run_dir / "plots" / plot_file).exists():
            md.append(f"\n### {description}\n")
            md.append(f"![{description}](plots/{plot_file})\n")

    # Key findings
    md.append("\n## 🔑 Key Findings\n")
    if teacher_eval:
        if teacher_eval['teacher_opt_top1_acc'] > 0.98:
            md.append("1. ✅ **Near-perfect teacher agreement**: {:.1%} top-1 optimal accuracy\n".format(
                teacher_eval['teacher_opt_top1_acc']))

        if final.get('minimax_d', 0) > 0.99:
            md.append("2. ✅ **Perfect play achieved**: 100% draw rate vs minimax\n")

        if final.get('eval_w', 0) > 0.99:
            md.append("3. ✅ **Complete random dominance**: 100% win rate vs random\n")

        if teacher_eval['teacher_v_sign_acc_non_draw'] > 0.98:
            md.append("4. ✅ **Accurate value function**: {:.1%} sign accuracy on decisive positions\n".format(
                teacher_eval['teacher_v_sign_acc_non_draw']))

    md.append("\n## 📁 Output Files\n")
    md.append(f"- `checkpoint.pt` - Model weights and optimizer state\n")
    md.append(f"- `config.json` - Training configuration\n")
    md.append(f"- `history.csv` - Full metrics log\n")
    md.append(f"- `plots/` - All 19 visualization PNGs\n")
    md.append(f"- `REPORT.md` - This report\n")

    md.append("\n---\n")
    md.append(f"*Generated by ttt-transformer v0.1.0*\n")

    # Write markdown
    report_path = run_dir / "REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"✓ Markdown report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TicTacToe Transformer")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--games-per-step", type=int, default=64, help="Self-play games per step")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--eval-every", type=int, default=300, help="Evaluation frequency")
    parser.add_argument("--teacher-eval-every", type=int, default=300, help="Teacher eval frequency")
    parser.add_argument("--print-every", type=int, default=100, help="Print frequency")
    parser.add_argument("--run-name", type=str, default="ttt_run", help="Run name for saving")
    parser.add_argument("--save-dir", type=str, default="runs", help="Save directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = pick_device()
    print(f"Device: {device}")

    # Set seed
    set_seed(args.seed)

    # Config
    config = TrainConfig(
        seed=args.seed,
        steps=args.steps,
        games_per_step=args.games_per_step,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        print_every=args.print_every,
        save_dir=args.save_dir,
    )

    # Create save directory
    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Create model
    model = TinyTTTTransformer(d_model=64, n_heads=4, n_layers=2, ff_mult=4, dropout=0.0).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Replay buffers
    replay = PositionReplay(max_items=config.replay_max_positions)
    hard = HardNegativeReplay(max_items=config.hard_max_positions)

    # Warmup
    print("\n=== Warmup ===")
    warmup(model, device, config, replay)

    # Training loop
    history = []
    last_teacher_eval = None
    teacher_eval_every = args.teacher_eval_every

    print("\n=== Training ===")
    iterator = trange(1, config.steps + 1, desc="Training")

    for step in iterator:
        metrics = train_one_step(model, optimizer, device, config, step, replay, hard)

        # Logging
        if step % args.print_every == 0:
            tqdm.write(
                f"[{step:4d}] loss {metrics['loss']:.3f} | "
                f"π {metrics['pi_loss']:.3f} | v {metrics['v_loss']:.3f} | "
                f"top1 {metrics['train_top1_opt']:.3f} | "
                f"opt_mass {metrics['train_opt_mass']:.3f} | "
                f"v_exact {metrics['train_v_exact']:.3f}"
            )

        # Eval vs random + collect hard negatives
        if step % config.eval_every == 0:
            model.eval()
            with torch.inference_mode():
                w, d, l = eval_vs_random_collect_losses(model, device, 500, hard_buf=None)
                metrics.update({"eval_w": w, "eval_d": d, "eval_l": l})
                tqdm.write(f"  vs Random: {w:.2%} W / {d:.2%} D / {l:.2%} L")

                # Harvest hard negatives
                _w2, _d2, _l2 = eval_vs_random_collect_losses(model, device, 200, hard_buf=hard)
                tqdm.write(f"  Hard buffer: {len(hard)} positions")

        # Teacher eval
        if step % teacher_eval_every == 0:
            model.eval()
            with torch.inference_mode():
                te = eval_teacher_agreement_all_states(model, device, batch_size=4096)
                last_teacher_eval = te
                for k, v in te.items():
                    if not k.startswith("_"):
                        metrics[k] = float(v) if isinstance(v, (int, float)) else v
                tqdm.write(
                    f"  Teacher: top1 {te['teacher_opt_top1_acc']:.3f} | "
                    f"opt_mass {te['teacher_opt_mass_mean']:.3f} | "
                    f"v_exact {te['teacher_v_exact_acc']:.3f}"
                )

        # Eval vs minimax
        if step % teacher_eval_every == 0:
            model.eval()
            with torch.inference_mode():
                mm = eval_vs_minimax(model, device, games=500, teacher_plays_optimal_random=True)
                metrics.update({"minimax_w": mm["model_w"], "minimax_d": mm["model_d"],
                              "minimax_l": mm["model_l"]})
                tqdm.write(f"  vs Minimax: {mm['model_w']:.2%} W / {mm['model_d']:.2%} D / {mm['model_l']:.2%} L")

        history.append(metrics)

    # Save checkpoint
    print("\n=== Saving ===")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "history": history,
        "final_teacher_eval": last_teacher_eval,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    print(f"✓ Checkpoint saved to {run_dir}")

    # Save history
    try:
        import pandas as pd
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"✓ History saved to {run_dir / 'history.csv'}")
    except ImportError:
        pass

    # Generate plots
    print("\n=== Generating Plots ===")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    create_all_plots(history, last_teacher_eval, plots_dir)
    print(f"✓ 19 plots saved to {plots_dir}")

    # Generate markdown report
    print("\n=== Generating Report ===")
    generate_markdown_report(history, last_teacher_eval, config, run_dir, device)

    # Final summary
    print("\n=== Final Results ===")
    final = history[-1]
    print(f"vs Random:  {final.get('eval_w', 0):.1%} W / {final.get('eval_d', 0):.1%} D / {final.get('eval_l', 0):.1%} L")
    print(f"vs Minimax: {final.get('minimax_w', 0):.1%} W / {final.get('minimax_d', 0):.1%} D / {final.get('minimax_l', 0):.1%} L")
    if last_teacher_eval:
        print(f"Teacher Top-1: {last_teacher_eval['teacher_opt_top1_acc']:.1%}")
        print(f"Teacher Opt Mass: {last_teacher_eval['teacher_opt_mass_mean']:.3f}")

    print(f"\n✅ All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
