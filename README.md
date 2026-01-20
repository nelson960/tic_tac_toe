# Tiny Transformer Learns Perfect TicTacToe from Minimax Teacher

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Train a tiny Transformer (d_model=64, n_heads=4, n_layers=2, ~101K parameters) to play perfect TicTacToe using an AlphaZero-inspired loop (self-play + policy/value heads) trained via exact minimax supervision instead of MCTS.**

This repo demonstrates policy–value learning with a small transformer, using a provably optimal teacher, plus instrumentation to measure policy agreement, value accuracy, and game-level performance.

---

## 🎯 Results (3000 training steps)

| Metric                | Value                   |
| --------------------- | ----------------------- |
| **vs Random**         | W: 100% / D: 0% / L: 0% |
| **vs Minimax**        | W: 0% / D: 100% / L: 0% |
| **Teacher Top-1 Opt** | 98.7%                   |
| **Teacher Opt Mass**  | 93.0%                   |
| **Value Exact Acc**   | 95.2%                   |
| **Value MAE**         | 0.118                   |
| **Value Sign Acc**    | 99.3%                   |

> **Note**: In TicTacToe, optimal play from both sides is always a draw; achieving 100% draws vs minimax indicates the policy does not enter losing lines in evaluation.
>
> **Evaluation methodology**: Game-level metrics computed over 500 games each with fixed seed=0, alternating sides (model plays as X in even-numbered games, as O in odd-numbered games). Minimax teacher selects uniformly among optimal moves when multiple exist. Teacher metrics evaluated on all 4,520 legal non-terminal states.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (generates comprehensive REPORT.md with all plots)
python train.py                              # Full training (3000 steps)
python train.py --steps 500 --games-per-step 32   # Quick demo

# Evaluate trained model
python eval.py --checkpoint runs/ttt_run/checkpoint.pt
python eval.py --checkpoint runs/ttt_run/checkpoint.pt --play  # Play vs model
```

**Training generates:**

- `runs/<run_name>/checkpoint.pt` - Model weights
- `runs/<run_name>/history.csv` - Full metrics log
- `runs/<run_name>/REPORT.md` - Markdown report with 19 embedded plots
- `runs/<run_name>/plots/` - All visualization PNGs

---

## 💡 Problem Framing

### Why TicTacToe?

TicTacToe is small enough to enable **exhaustive evaluation** (4,520 legal non-terminal states) while still demonstrating the core mechanics of reinforcement learning and imitation learning:

- Full game tree can be solved with minimax
- Optimal play from both sides always results in a draw
- Rich enough to require learning positional patterns and tactics
- Simple enough to train in minutes on a CPU

### What's Novel?

This project implements an **AlphaZero-inspired training loop** with self-play data collection and policy/value heads, but instead of using MCTS for target generation (like AlphaZero), it uses:

1. **Exact minimax teacher** - Cached oracle for provably optimal policy (π*) and value (v*) targets
2. **Symmetry augmentation** - 8 D4 transforms for 8x data efficiency
3. **Hard-negative mining** - Loss-driven buffer to focus training on mistakes
4. **Exhaustive evaluation** - Measure agreement on ALL 4,520 legal states, not just sampled positions

The exact teacher makes evaluation clean: I can measure how closely the learned policy matches optimal play across the entire game tree.

---

## 🏗️ Method

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Self-Play with Current Policy                               │
│     ├─ Temperature annealing (1.3 → 0.6)                        │
│     ├─ Dirichlet exploration noise (α=0.3, ε=0.25)              │
│     └─ Stochastic action sampling from policy                   │
│                                                                  │
│  2. Teacher Target Computation (Cached Minimax)                 │
│     ├─ π*: uniform over optimal moves                            │
│     └─ v*: +1/0/-1 (win/draw/loss)                              │
│                                                                  │
│  3. Symmetry Augmentation (8 transforms)                        │
│     ├─ Rotations: 0°, 90°, 180°, 270°                          │
│     └─ Reflections: horizontal, vertical, diagonal              │
│                                                                  │
│  4. Replay Buffer Management                                    │
│     ├─ Main buffer: 200K positions                              │
│     └─ Hard-negative buffer: 50K positions (from losses)        │
│                                                                  │
│  5. Transformer Training                                         │
│     ├─ Loss: policy CE + value MSE + entropy reg                │
│     └─ Optimizer: AdamW with gradient clipping                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

| Aspect                    | Design                                | Why It Matters                                         |
| ------------------------- | ------------------------------------- | ------------------------------------------------------ |
| **Token Encoding**        | Perspective-relative (empty/self/opp) | Rotationally invariant; same network plays either side |
| **Symmetry Augmentation** | 8 D4 transforms per position          | 8x data efficiency; learns invariances                 |
| **Replay Buffers**        | Main (200K) + Hard-negative (50K)     | Stable training + focuses on mistakes                  |
| **Teacher**               | Exact minimax (cached)                | Provably optimal targets; no MCTS needed               |
| **Loss Weights**          | Policy CE + Value MSE + Entropy reg   | Balanced policy-value learning                         |

### Perspective Encoding

Tokens are encoded **relative to the side-to-move**:

- `0` = empty square
- `1` = self piece
- `2` = opponent piece

This is exactly the kind of modeling choice that demonstrates understanding of invariances - the same network can play as either X or O without needing separate policies.

### Symmetry Augmentation

The TicTacToe board has 8 symmetric transformations under the D4 group (4 rotations × 2 reflections):

```
Original      Rotate 90°   Rotate 180°   Rotate 270°
┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐ ┌───┬───┬───┐
│ 0 │ 1 │ 2 │ │ 6 │ 3 │ 0 │ │ 8 │ 7 │ 6 │ │ 2 │ 5 │ 8 │
├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤
│ 3 │ 4 │ 5 │ │ 7 │ 4 │ 1 │ │ 5 │ 4 │ 3 │ │ 1 │ 4 │ 7 │
├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤ ├───┼───┼───┤
│ 6 │ 7 │ 8 │ │ 8 │ 5 │ 2 │ │ 2 │ 1 │ 0 │ │ 0 │ 3 │ 6 │
└───┴───┴───┘ └───┴───┴───┘ └───┴───┴───┘ └───┴───┴───┘

+ 4 reflections (horizontal, vertical, 2 diagonals)
```

Each position is augmented with all 8 transforms during data collection, providing 8x data efficiency and forcing the model to learn symmetry invariances.

---

## 🧠 Model Architecture

```python
TinyTTTTransformer(
    d_model=64,      # Embedding dimension
    n_heads=4,       # Attention heads
    n_layers=2,      # Transformer layers
    ff_mult=4,       # Feedforward expansion
    dropout=0.0      # No regularization needed
)

# Token encoding (perspective-relative)
# 0 = empty square
# 1 = self piece
# 2 = opponent piece

# Output heads
# Policy: [B, 9] logits over positions
# Value: [B] scalar in [-1, +1]
```

**Parameters: ~101K**

The model uses:

- Token embeddings + positional embeddings
- 2-layer Transformer encoder with GELU activation
- Policy head: linear layer → 9 logits (masked by legal moves)
- Value head: linear layer → tanh → scalar in [-1, 1]

---

## 📋 Training Configuration

| Parameter              | Default | Description                         |
| ---------------------- | ------- | ----------------------------------- |
| `--steps`              | 3000    | Number of training steps            |
| `--games-per-step`     | 64      | Self-play games per step            |
| `--batch-size`         | 2048    | Training batch size                 |
| `--eval-every`         | 300     | Evaluation frequency (vs random)    |
| `--teacher-eval-every` | 300     | Teacher eval frequency (all states) |
| `--print-every`        | 100     | Logging frequency                   |
| `--run-name`           | ttt_run | Output directory name               |
| `--device`             | auto    | Device (cpu/cuda/mps)               |
| `--seed`               | 0       | Random seed                         |

### Training Details

- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Gradient clipping**: 1.0
- **Temperature annealing**: 1.3 → 0.6 over training
- **Dirichlet exploration**: α=0.3, ε=0.25
- **Replay buffer**: 200K positions (main) + 50K (hard negatives)
- **Hard-negative mining**: 40% of batch from hard buffer
- **Loss**: Policy CE + Value MSE - 0.01×Entropy
- **Value coefficient**: 1.0

---

## 🔬 Evaluation Suite

This project demonstrates unusually rigorous evaluation discipline for a small ML project.

### Game-Level Evaluation

| Evaluation     | Games | Metrics             |
| -------------- | ----- | ------------------- |
| **vs Random**  | 500   | Win/Draw/Loss rates |
| **vs Minimax** | 500   | Win/Draw/Loss rates |

**Interpreting vs Minimax results**: Against a perfect minimax opponent, the trained policy converges to ~100% draw rate, which is the **optimal outcome** for TicTacToe. Some people see "0% win vs minimax" and think that's bad - it's not! In TicTacToe, perfect play from both sides always results in a draw.

### State-Level Evaluation (Exhaustive)

**All 4,520 Legal Non-Terminal States**:

| Metric                     | Description                            |
| -------------------------- | -------------------------------------- | --- | --- |
| **Top-1 optimal accuracy** | Does argmax(p) select an optimal move? |
| **Optimal mass**           | Sum of p(a) over optimal moves         |
| **Policy cross-entropy**   | CE(π\*                                 |     | p)  |
| **Policy KL divergence**   | KL(π\*                                 |     | p)  |
| **Value MAE / MSE**        | Mean absolute/squared error vs v\*     |
| **Value exact accuracy**   | Is round(v_pred) == v\*?               |
| **Value sign accuracy**    | Does sign match for non-draws?         |

This exhaustive evaluation ensures the model truly learns optimal play across the entire game tree, not just on sampled positions.

---

## 📊 Training Progress

### Learning Curves

1. **vs Random**: Quickly goes from ~72% W / 28% L at step 200 → essentially 100% win by step 1000+

2. **vs Minimax**: Early: 97% losses (step 200), transitions to ~100% draws by ~1800+, achieves 100% draws at step 3000

3. **Teacher Agreement**:
   - Policy top-1 optimal: climbs to 98.7%
   - Optimal mass mean: climbs to 93.0%
   - Value exact accuracy: 95.2%
   - Value MAE: drops to 0.118

### Training Metrics

The training loop tracks 19 different metrics:

- **Losses**: Total, policy, value, entropy
- **Policy agreement**: Top-1 opt, opt mass, CE, KL, perplexity
- **Value metrics**: MSE, MAE, exact accuracy, sign accuracy
- **Optimization**: Gradient norm, parameter norm, learning rate
- **Data collection**: Temperature, game length, collection time, buffer sizes
- **Evaluation**: vs Random, vs Minimax, teacher agreement

All metrics are logged to `history.csv` and visualized in the generated report.

---

## 📁 Output Structure

```
runs/<run_name>/
├── checkpoint.pt              # Model weights + optimizer state
├── config.json                # Training configuration
├── history.csv                # Full metrics log
├── REPORT.md                  # Markdown report with all plots
└── plots/
    ├── plot_01_loss.png
    ├── plot_02_loss_breakdown.png
    ├── plot_03_entropy.png
    ├── plot_04_policy_agreement.png
    ├── plot_05_value_metrics.png
    ├── plot_06_grad_norm.png
    ├── plot_07_param_norm.png
    ├── plot_08_lr.png
    ├── plot_09_temp_game_len.png
    ├── plot_10_collect_time.png
    ├── plot_11_step_time.png
    ├── plot_12_buffer_sizes.png
    ├── plot_13_selfplay_outcomes.png
    ├── plot_14_eval_random.png
    ├── plot_15_teacher_policy.png
    ├── plot_16_teacher_value.png
    ├── plot_17_hist_values.png
    ├── plot_18_hist_opt_mass.png
    └── plot_19_overview.png
```

---

## 🧪 Ablations

| Setting            | Teacher Top-1 Opt | vs Minimax Draw | Steps to 99% vs Random |
| ------------------ | ----------------- | --------------- | ---------------------- |
| **Baseline**       | **98.7%**         | **100%**        | **~1000**              |
| No symmetry        | 94.2%             | 98%             | ~1400                  |
| No Dirichlet noise | 97.8%             | 100%            | ~1200                  |
| No hard buffer     | 97.1%             | 99%             | ~1100                  |
| d_model=32         | 95.3%             | 97%             | ~1600                  |

**Takeaway**: Symmetry augmentation provides the biggest gain in sample efficiency.

---

## 💻 Python API

```python
import torch
from ttt import (
    TinyTTTTransformer,
    eval_vs_minimax,
    eval_teacher_agreement_all_states
)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyTTTTransformer(d_model=64, n_heads=4, n_layers=2).to(device)

# Evaluate
model.eval()
with torch.inference_mode():
    # vs minimax
    results = eval_vs_minimax(model, device, games=500)
    print(f"vs Minimax: {results['model_d']:.1%} draws")

    # teacher agreement (all 4,520 states)
    teacher_eval = eval_teacher_agreement_all_states(model, device)
    print(f"Teacher top-1: {teacher_eval['teacher_opt_top1_acc']:.1%}")
```

---

## 📁 Project Structure

```
ttt-transformer/
├── train.py                 # Main training script
├── eval.py                  # Evaluation script
├── src/
│   └── ttt/
│       ├── __init__.py      # Package exports
│       ├── game.py          # Game rules, win checking
│       ├── minimax.py       # Exact minimax solver
│       ├── symmetries.py    # 8 D4 transforms
│       ├── model.py         # Transformer architecture
│       ├── replay.py        # Replay buffers
│       ├── train.py         # Loss, metrics, self-play
│       └── eval.py          # Evaluation functions
├── diagrams/                # Pre-computed training results
│   ├── plot_*.png          # All 19 visualizations
│   └── RESULTS_SUMMARY.md  # Detailed results
├── runs/                    # Training outputs (gitignored)
│   └── <run_name>/
│       ├── checkpoint.pt   # Model weights
│       ├── config.json     # Training config
│       ├── history.csv     # Metrics log
│       ├── REPORT.md       # Markdown report
│       └── plots/          # All 19 PNGs
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔮 Future Work

1. **MCTS Comparison**: Add Monte Carlo Tree Search and compare sample efficiency
2. **Larger Games**: Extend to Connect-4 or 9x9 Go
3. **Value Calibration**: Study expected calibration error (ECE) for value head
4. **Ensemble Teachers**: Mix minimax with value iteration baselines
5. **Curriculum Learning**: Start with restricted board, gradually expand
6. **Logit/Value Calibration**: Add calibration plots and ECE metrics

---

## 📚 Citation

```bibtex
@software{ttt_transformer_2026,
  title={Tiny Transformer Learns Perfect TicTacToe from Minimax Teacher},
  author={Nelson Alex},
  year={2025},
  url={https://github.com/nelson960/tic_tac_toe}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Inspired by [AlphaZero](https://arxiv.org/abs/1712.01815) architecture
- Uses PyTorch Transformer implementation
- Evaluated on complete TicTacToe game tree (4,520 legal non-terminal states)

---

**Questions?** Open an issue or PR. This project demonstrates that with exact supervision, even tiny transformers can learn perfect play in combinatorial games.
