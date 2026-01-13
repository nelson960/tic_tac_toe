# Tic-Tac-Toe Transformer (Self-Play)

This project trains a tiny Transformer to play Tic-Tac-Toe **from scratch** using **self-play** (no dataset).

## What the code does

### 1) Game environment
A simple Tic-Tac-Toe environment:
- board is 9 cells: `0` empty, `+1` X, `-1` O
- `legal_moves()` returns empty squares
- `is_terminal()` checks win/draw
- `step(action)` applies a move and flips the player

### 2) Board encoding (perspective tokens)
Before giving the board to the model, it’s converted into **tokens from the current player’s perspective**:

- `0` = empty
- `1` = **my** piece
- `2` = **opponent** piece

This means the model always sees “me vs opponent” the same way, whether it is playing X or O.

### 3) Model: Tiny Transformer (policy + value)
The Transformer reads the 9 tokens (one per square) and outputs:

- **policy logits**: 9 scores (one per possible move)
- **value**: a single number in `[-1, +1]` (expected outcome for the current player)

Position embeddings are added so the model knows which square is which.

### 4) Self-play (data generation)
Training data is created by letting the model play against itself:
- for each move, it samples an action using the policy (with masking so illegal moves can’t be chosen)
- exploration is added using `epsilon` (random legal move sometimes) and `temperature` (softens sampling)

A faster version plays many games in parallel (vectorized).

At the end of each game, every move gets a label:
- `z = +1` if that move belonged to the winning side
- `z = -1` if it belonged to the losing side
- `z = 0` for draws

### 5) Training (policy + value learning)
For all collected states in a batch:
- **policy loss** pushes the model to increase probability of moves that led to wins and decrease moves that led to losses
- **value loss** trains the value head to predict `z`
- **entropy bonus** keeps the policy from collapsing too early (encourages exploration)

### 6) Evaluation
Every `eval_every` steps, the model plays games vs a random opponent and reports win/draw/loss rate.

