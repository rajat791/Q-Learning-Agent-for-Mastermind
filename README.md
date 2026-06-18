# Q-Learning for Mastermind

A tabular Q-learning agent trained to solve the Mastermind code-breaking game. This project investigates the effect of state representation and exploration strategy on reinforcement learning performance, with all results benchmarked against a random baseline.

---

## Overview

Mastermind is a code-breaking game where a player attempts to guess a secret code within a fixed number of turns. After each guess, feedback is given in the form of black pegs (correct colour, correct position) and white pegs (correct colour, wrong position).

This project models Mastermind as a single-agent reinforcement learning problem and trains a tabular Q-learning agent to learn an effective guessing policy. Three experiments are conducted:

1. **History length** — how much past feedback the agent can see
2. **Exploration rate (epsilon)** — how often the agent explores vs exploits
3. **Learning curves** — how quickly the agent converges

---

## Game Configuration

| Parameter | Value |
|---|---|
| Number of colours | 4 |
| Number of positions | 2 |
| Maximum turns | 6 |
| Possible codes | 16 |
| Win reward | +10 |
| Step penalty | -1 |

---

## Method

### Q-Learning

The agent learns an action-value function Q(s, a) using the standard temporal-difference update rule:

```
Q(s, a) ← Q(s, a) + η * [target - Q(s, a)]
```

Where the target is:
- `r` if the state is terminal
- `r + γ * max_a' Q(s', a')` otherwise

Fixed hyperparameters: `η = 0.2`, `γ = 0.99`

### State Representation

The state is a tuple of the last `history_length` turn records. Each record contains `(guess_pos1, guess_pos2, black_pegs, white_pegs)`. Empty slots are filled with `None`.

### Epsilon-Greedy Exploration

During training, the agent selects a random action with probability ε (exploration) and the greedy best action with probability 1-ε (exploitation). During evaluation, ε is set to 0.

---

## Experiment Results

### Experiment 1 — History Length

| History Length | Win Rate (%) | Avg Turns |
|---|---|---|
| 1 | 81.9 ± 13.4 | 3.92 ± 0.44 |
| 2 | 100.0 ± 0.0 | 3.02 ± 0.18 |
| 3 | 100.0 ± 0.0 | 2.92 ± 0.16 |
| 4 | 100.0 ± 0.0 | 2.89 ± 0.10 |
| Random baseline | 34.3 | 5.74 |

**Key finding:** History length 2 is optimal — it achieves 100% win rate with the smallest state space. A history of 1 fails to satisfy the Markov property, giving the agent insufficient context to rule out impossible codes.

### Experiment 2 — Exploration Rate (ε)

| Epsilon | Win Rate (%) | Avg Turns |
|---|---|---|
| 0.0 | 100.0 ± 0.0 | 3.03 ± 0.25 |
| 0.1 | 100.0 ± 0.0 | 3.16 ± 0.22 |
| 0.2 | 100.0 ± 0.0 | 3.07 ± 0.18 |
| 0.3 | 100.0 ± 0.0 | 2.96 ± 0.14 |
| 0.5 | 98.5 ± 4.4 | 3.06 ± 0.23 |
| Random baseline | 34.8 | 5.72 |

**Key finding:** ε = 0.3 produced the lowest average turns (2.96) and the most stable policy (±0.14 std). ε = 0 still achieved 100% win rate due to implicit exploration from random tie-breaking on unseen states. ε = 0.5 caused slight degradation.

### Experiment 3 — Learning Curves

- The agent reached 100% win rate within the first 2,000 training episodes
- Performance plateaued well before the 10,000 episode budget
- Mean ± std bands across 10 independent runs showed stable convergence

**Final configuration:** `history_length = 2`, `ε = 0.3`

---

## Key Findings

- A history length of at least 2 is needed to make the state effectively Markovian — with history 1, the agent cannot tell which turn it is on or recall earlier feedback
- Moderate exploration (ε = 0.3) gave the most efficient and consistent policy
- The agent converged within 2,000 episodes, far below the 10,000 episode training budget
- All tested configurations significantly outperformed the random baseline (34.3% win rate)

---

## Project Structure

```
├── environment.py        # MastermindEnv (provided)
├── qtable.py             # QTable implementation (provided)
├── solution.py           # QLearningAgent + experiments
├── history_length.png    # Experiment 1 plot
├── epsilon.png           # Experiment 2 plot
├── learning_curves.png   # Experiment 3 plot
└── README.md
```

---

## How to Run

```bash
pip install numpy matplotlib
python solution.py
```

The script runs the implementation check first, then the three experiments in sequence.

---

## Technologies

- Python
- NumPy
- Matplotlib
