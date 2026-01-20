#!/usr/bin/env python3
"""
Evaluate a trained TicTacToe model.

Usage:
    python eval.py --checkpoint runs/ttt_run/checkpoint.pt
    python eval.py --checkpoint runs/ttt_run/checkpoint.pt --play
"""

import sys
import argparse
from pathlib import Path

import torch

# Add src to path
sys.path = [str(Path(__file__).parent / "src")] + sys.path

from ttt import (
    TinyTTTTransformer,
    eval_vs_random,
    eval_vs_minimax,
    eval_teacher_agreement_all_states,
)


def print_board(board):
    """Pretty print board."""
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    for i in range(3):
        row = "|".join(symbols[board[i*3 + j]] for j in range(3))
        print(row)
        if i < 2:
            print("-+-+-")


def play_interactive(model, device):
    """Play a game against the model."""
    from ttt import is_terminal, legal_moves, apply_move, board_to_tokens_perspective, legal_move_mask

    board = [0] * 9
    player = +1

    print("\n=== Interactive Game ===")
    print("You are X (play first)")
    print("Enter moves as numbers 0-8:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    print()

    while True:
        done, winner = is_terminal(board)
        if done:
            print_board(board)
            if winner == 0:
                print("\nDraw!")
            elif winner == +1:
                print("\nYou win!")
            else:
                print("\nModel wins!")
            break

        print_board(board)
        print()

        if player == +1:
            moves = legal_moves(board)
            try:
                action = int(input(f"Your move ({moves}): "))
                if action not in moves:
                    print("Invalid move, try again")
                    continue
            except (ValueError, KeyboardInterrupt):
                print("\nGame aborted")
                return
        else:
            tokens = board_to_tokens_perspective(board, player).to(device).unsqueeze(0)
            mask = legal_move_mask(board).to(device).unsqueeze(0)
            logits, _ = model(tokens)
            action = int(logits.squeeze(0).argmax().item())
            print(f"Model plays: {action}")

        board[action] = player
        player = -player
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TicTacToe model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--play", action="store_true", help="Play interactive game")
    parser.add_argument("--games", type=int, default=100, help="Number of eval games")

    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model
    model = TinyTTTTransformer(d_model=64, n_heads=4, n_layers=2, ff_mult=4, dropout=0.0)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Interactive play
    if args.play:
        with torch.inference_mode():
            play_interactive(model, device)
        return

    # Evaluation
    print("\n=== Evaluation ===")
    with torch.inference_mode():
        # vs Random
        print(f"\nvs Random ({args.games} games)...")
        w, d, l = eval_vs_random(model, device, games=args.games)
        print(f"  Wins:   {w:.2%}")
        print(f"  Draws:  {d:.2%}")
        print(f"  Losses: {l:.2%}")

        # vs Minimax
        print(f"\nvs Minimax ({args.games} games)...")
        results = eval_vs_minimax(model, device, games=args.games)
        print(f"  Wins:   {results['model_w']:.2%}")
        print(f"  Draws:  {results['model_d']:.2%}")
        print(f"  Losses: {results['model_l']:.2%}")

        # Teacher agreement
        print(f"\nTeacher Agreement (all states)...")
        te = eval_teacher_agreement_all_states(model, device, batch_size=4096)
        print(f"  States:      {te['teacher_n_states']}")
        print(f"  Top-1 Opt:   {te['teacher_opt_top1_acc']:.2%}")
        print(f"  Opt Mass:    {te['teacher_opt_mass_mean']:.3f}")
        print(f"  Value Exact: {te['teacher_v_exact_acc']:.2%}")
        print(f"  Value MAE:   {te['teacher_v_mae_mean']:.4f}")


if __name__ == "__main__":
    main()
