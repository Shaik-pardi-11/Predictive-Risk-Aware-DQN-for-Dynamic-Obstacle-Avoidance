#  Risk-Aware DQN — Dynamic Obstacle Avoidance in Maze Navigation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Algorithm-Deep%20Q--Network-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Environment-Custom%20OpenAI%20Gym-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Research%20Project-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

> A reinforcement learning project comparing **Standard DQN** vs **Risk-Aware DQN** for autonomous navigation in dynamic environments with moving obstacles. The risk-aware agent uses trajectory prediction and a custom reward function to make **proactive** safety decisions — avoiding danger before it arrives, not after.

---

##  Table of Contents

- [Overview](#-overview)
- [Key Idea](#-key-idea)
- [Project Structure](#-project-structure)
- [Environment](#-environment)
- [Agents](#-agents)
- [Reward Design](#-reward-design)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Future Work](#-future-work)

---

## Overview

Most reinforcement learning navigation agents are **reactive** — they learn to avoid obstacles only when already close to them. In fast-moving dynamic environments, this is often too late.

This project implements a **Risk-Aware Deep Q-Network (RA-DQN)** that:
- Predicts future obstacle positions using a trajectory predictor
- Builds a probabilistic **risk map** of the environment at every step
- Uses a custom multi-component reward function that **penalises risky future paths**
- Is compared directly against a Standard DQN baseline under identical conditions

---

##  Key Idea

| Agent | Strategy | Behaviour |
|---|---|---|
| Standard DQN | Reactive | Avoids obstacles when already adjacent |
| Risk-Aware DQN | **Proactive** | Predicts obstacle paths and avoids high-risk areas in advance |

The core innovation is the **risk map** — a real-time grid showing predicted obstacle occupancy probability — which is included in the agent's state observation and used to shape rewards.

---

## Project Structure

```
├── env.py              # Custom dynamic maze environment (OpenAI Gym-style)
├── agent.py            # StandardDQNAgent and RiskAwareDQNAgent (Dueling DQN + PER)
├── train.py            # Experiment runner — trains both agents for 500 episodes
├── evaluate.py         # Policy evaluator and safety metrics
├── riskanalysis.py     # Trajectory predictor, risk mapper, safety metrics
├── test_penalty.py     # Ablation study — penalty weight configurations
├── dashboard.html      # Interactive training results dashboard
├── results.json        # Saved training results (500 episodes)
└── TRAINING_ANALYSIS.md  # Notes on reward structure and training behaviour
```

---

##  Environment

The environment (`env.py`) is a custom **12×12 grid-based maze** with:

- **Agent** — navigates from a random start to a random goal
- **Dynamic obstacles** — move continuously with bouncing boundary behaviour
- **Risk map** — computed at every step using obstacle trajectory prediction
- **Observation space** — includes agent position, goal direction, obstacle states, full risk map, and predicted future obstacle positions

### State Vector Composition

```
s_t = [
  agent_x, agent_y,                        # 2  — agent position (normalised)
  goal_dx, goal_dy,                         # 2  — direction to goal
  obs_x, obs_y, obs_vx, obs_vy × N,        # 4N — obstacle states
  risk_map (flattened),                     # G² — full risk grid
  predicted_positions × N × H              # 2NH — future obstacle positions
]
```

Where `N = 5` obstacles, `G = 12` grid size, `H = 5` risk horizon.
**Total observation dimension: 244**

### Environment Parameters

| Parameter | Value | Description |
|---|---|---|
| Grid size | 12 × 12 | Maze dimensions |
| Obstacles | 5 | Dynamic moving obstacles |
| Max steps | 150 | Per episode limit |
| Risk horizon | 5 | Steps ahead for trajectory prediction |
| Min start-goal distance | 6 (Manhattan) | Ensures non-trivial episodes |

---

##  Agents

Both agents share the same **Dueling DQN** architecture with **Prioritised Experience Replay (PER)** and **Double DQN** updates.

### Neural Network — Dueling DQN

```
Input  (244)
  │
  ├─ Shared Layers: Dense(256, ReLU) → Dense(128, ReLU)
  │
  ├─ Value Stream:  Dense(128, ReLU) → Dense(1, Linear)        → V(s)
  └─ Advantage:     Dense(128, ReLU) → Dense(5, Linear)        → A(s,a)

Output = V(s) + A(s,a) - mean(A(s,a))     # 5 Q-values (UP/DOWN/LEFT/RIGHT/STAY)
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Discount factor γ | 0.99 |
| Epsilon start → end | 1.0 → 0.05 |
| Epsilon decay steps | 8,000 |
| Batch size | 64 |
| Replay buffer capacity | 30,000 |
| Target network update | Every 100 steps (soft update τ=0.005) |
| Learning rate | 0.001 |
| Hidden dimensions | [256, 128] |

### `StandardDQNAgent`
Vanilla Dueling DQN with PER. No explicit risk awareness in action selection or reward.

### `RiskAwareDQNAgent`
Same architecture as Standard, but trained on the risk-shaped reward signal. The risk map is part of the state, making the agent implicitly aware of future danger.

---

##  Reward Design

The reward function is the core innovation of this project. It has three components:

### Base Rewards

| Event | Reward |
|---|---|
| Goal reached | `+50.0` |
| Collision with obstacle | `-20.0` |
| Timeout (max steps) | `-5.0` |
| Step penalty | `-0.5` per step |
| Progress toward goal | `+(prev_dist - curr_dist) × 1.5` |

### Risk Penalty (Risk-Aware Agent)

```python
reward -= risk_map[agent_x, agent_y] * 3.0
```

The risk map value at the agent's current cell (range 0–1) is multiplied by a weight of **3.0** and subtracted from the reward. This penalises the agent for occupying high-risk areas — even if no collision has occurred yet — encouraging it to prefer safer routes proactively.

### Typical Episode Breakdown

```
~150 steps without goal:
  -0.5 × 150   =  -75.0   (step penalties)
  -20.0                    (collision penalty)
  -20 to -50               (risk penalties)
  +0 to +10                (progress reward)
  ─────────────────────
  = -115 to -155           (expected range)
```

---

##  Results

Training was conducted for **500 episodes** on a 12×12 grid with 5 dynamic obstacles.

### Final Performance (last 50 episodes)

| Metric | Standard DQN | Risk-Aware DQN | Improvement |
|---|---|---|---|
| Avg Reward | -34.07 | -32.85 | **+1.22** |
| Collision Rate | 1.00 | 0.98 | **-2%** |
| Success Rate | 0.0% | **2.0%** | **+2%** |
| Total Goals (500 ep) | 3 | **6** | **2× more** |
| Total Collisions | 497 | **494** | **-3** |



### Ablation — Penalty Weight Study (`test_penalty.py`)

Four penalty configurations were tested (300 episodes each):

| Configuration | Step Penalty | Collision Penalty | Risk Weight |
|---|---|---|---|
| LOW | 0.2 | 10.0 | 1.0 |
| MEDIUM (Default) | 0.5 | 20.0 | 3.0 |
| HIGH | 1.0 | 30.0 | 5.0 |
| EXTREME | 2.0 | 50.0 | 7.0 |

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/risk-aware-dqn.git
cd risk-aware-dqn

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy
```

> **No external ML frameworks required.** The entire DQN implementation (neural network, backprop, Adam optimiser) is built from scratch using **NumPy only**.

---

##  Usage

### Train Both Agents (500 episodes)

```bash
python train.py
```

Output includes per-episode logs every 50 episodes and a final comparison table. Results are saved to `results.json`.

### Run Penalty Ablation Study

```bash
python test_penalty.py
```

Trains both agents across 4 penalty configurations (300 episodes each) and prints a comparison summary.

### View Training Dashboard

Open `dashboard.html` in any browser to view interactive charts of training metrics.

### Run Custom Experiment

```python
from train import ExperimentRunner

runner = ExperimentRunner(
    grid_size=12,
    num_obstacles=5,
    max_steps=150,
    risk_horizon=5,
    n_episodes=1000,
    seed=42
)
results = runner.run(verbose=True, log_interval=100)
runner.save_results("my_results.json")
```

### Evaluate a Trained Agent

```python
from evaluate import PolicyEvaluator
from env import DynamicNavEnvironment
from agent import RiskAwareDQNAgent

env = DynamicNavEnvironment(grid_size=12, num_obstacles=5, max_steps=150, risk_horizon=5)
agent = RiskAwareDQNAgent(state_dim=env.obs_dim, action_dim=env.action_space_n)

evaluator = PolicyEvaluator(env, agent, n_eval_episodes=100)
results = evaluator.evaluate()
```

### Render the Environment

```python
env = DynamicNavEnvironment()
env.reset()
print(env.render_ascii())
# Output:
# +------------+
# |............|
# |..A.........| ← Agent
# |....X.......| ← Obstacle
# |........G...| ← Goal
# +------------+
```

---

##  Architecture

### Risk Map Pipeline

```
Obstacles (position + velocity)
         │
         ▼
  TrajectoryPredictor          ← predicts next H positions per obstacle
         │
         ▼
     RiskMapper                ← Gaussian spread over predicted positions
         │                        with exponential time-decay (0.85^t)
         ▼
  Risk Map (12×12 grid)        ← values in [0, 1]
         │
    ┌────┴────┐
    │         │
 State obs   Reward penalty
 (flattened) (-risk × 3.0)
```

### Agent Training Loop

```
  Environment
      │  state s_t
      ▼
  Q-Network ──── ε-greedy ──── Action a_t
      │
      ▼
  Environment step → (s_{t+1}, r_t, done)
      │
      ▼
  Replay Buffer (PrioritizedReplayBuffer, cap=30k)
      │  sample batch (64)
      ▼
  Double DQN Update:
    best_a = argmax Q_online(s')
    target = r + γ · Q_target(s', best_a)
    loss   = MSE(Q_online(s,a), target)
      │
      ▼
  Soft update target network (τ=0.005) every 100 steps
```

---


##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---



---

<p align="center">Made with ❤️ for research in safe autonomous navigation</p>
