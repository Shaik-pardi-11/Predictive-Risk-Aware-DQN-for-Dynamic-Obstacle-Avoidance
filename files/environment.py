"""
Dynamic Navigation Environment
Implements a 2D grid world with moving obstacles for RL training.
Supports both reactive (standard DQN) and proactive (Risk-Aware DQN) modes.
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class Action(Enum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STAY  = 4


@dataclass
class Obstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.8
    trajectory_history: List[Tuple[float, float]] = field(default_factory=list)

    def update(self, grid_size: int) -> None:
        """Move obstacle and bounce off walls."""
        self.trajectory_history.append((self.x, self.y))
        if len(self.trajectory_history) > 20:
            self.trajectory_history.pop(0)

        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
        if self.x <= 0 or self.x >= grid_size - 1:
            self.vx *= -1
            self.x = np.clip(self.x, 0, grid_size - 1)
        if self.y <= 0 or self.y >= grid_size - 1:
            self.vy *= -1
            self.y = np.clip(self.y, 0, grid_size - 1)

    def predict_position(self, steps: int, grid_size: int) -> List[Tuple[float, float]]:
        """Predict future positions for risk assessment."""
        positions = []
        px, py = self.x, self.y
        pvx, pvy = self.vx, self.vy
        for _ in range(steps):
            px += pvx
            py += pvy
            if px <= 0 or px >= grid_size - 1:
                pvx *= -1
                px = np.clip(px, 0, grid_size - 1)
            if py <= 0 or py >= grid_size - 1:
                pvy *= -1
                py = np.clip(py, 0, grid_size - 1)
            positions.append((px, py))
        return positions


@dataclass
class NavState:
    agent_x: int
    agent_y: int
    goal_x: int
    goal_y: int
    obstacles: List[Obstacle]
    step: int = 0
    collision: bool = False
    reached_goal: bool = False
    risk_map: Optional[np.ndarray] = None


class DynamicNavEnvironment:
    """
    Grid-based navigation environment with dynamic obstacles.
    Supports risk-aware state representation for proactive DQN.
    """

    def __init__(
        self,
        grid_size: int = 12,
        num_obstacles: int = 5,
        max_steps: int = 200,
        risk_horizon: int = 5,
        seed: int = 42
    ):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.risk_horizon = risk_horizon
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # State/action spaces
        self.action_space_n = len(Action)
        self.obs_dim = self._compute_obs_dim()

        # Episode tracking
        self.state: Optional[NavState] = None
        self.episode_rewards: List[float] = []
        self.episode_collisions: List[int] = []
        self.episode_steps: List[int] = []

    def _compute_obs_dim(self) -> int:
        """
        State vector:
          - agent position (2)
          - goal relative position (2)
          - obstacle positions + velocities (4 * num_obstacles)
          - risk map flattened (grid_size^2) [proactive]
          - predicted obstacle positions horizon (2 * num_obstacles * risk_horizon) [proactive]
        """
        base = 2 + 2 + 4 * self.num_obstacles
        proactive = self.grid_size ** 2 + 2 * self.num_obstacles * self.risk_horizon
        return base + proactive

    def reset(self) -> np.ndarray:
        """Initialize a new episode."""
        gs = self.grid_size

        # Place agent at random edge or interior
        self.agent_x = self.rng.randint(1, gs - 2)
        self.agent_y = self.rng.randint(1, gs - 2)

        # Place goal opposite side
        self.goal_x = self.rng.randint(1, gs - 2)
        self.goal_y = self.rng.randint(1, gs - 2)
        while abs(self.goal_x - self.agent_x) + abs(self.goal_y - self.agent_y) < gs // 2:
            self.goal_x = self.rng.randint(1, gs - 2)
            self.goal_y = self.rng.randint(1, gs - 2)

        # Spawn obstacles with random velocities
        self.obstacles: List[Obstacle] = []
        for _ in range(self.num_obstacles):
            ox = self.rng.uniform(1, gs - 2)
            oy = self.rng.uniform(1, gs - 2)
            speed = self.rng.uniform(0.3, 0.8)
            angle = self.rng.uniform(0, 2 * np.pi)
            self.obstacles.append(Obstacle(
                x=ox, y=oy,
                vx=speed * np.cos(angle),
                vy=speed * np.sin(angle)
            ))

        self.step_count = 0
        self.collision_count = 0
        self.total_reward = 0.0
        self.prev_dist = self._manhattan_dist()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action, update world, compute reward."""
        assert self.state is None or True  # always live

        # Move agent
        dx, dy = 0, 0
        if action == Action.UP.value:    dy = -1
        elif action == Action.DOWN.value: dy = 1
        elif action == Action.LEFT.value:  dx = -1
        elif action == Action.RIGHT.value: dx = 1
        # STAY: dx=dy=0

        new_x = int(np.clip(self.agent_x + dx, 0, self.grid_size - 1))
        new_y = int(np.clip(self.agent_y + dy, 0, self.grid_size - 1))
        self.agent_x, self.agent_y = new_x, new_y

        # Update obstacles
        for obs in self.obstacles:
            obs.update(self.grid_size)

        self.step_count += 1
        reward, done, info = self._compute_reward()
        self.total_reward += reward

        if done:
            self.episode_rewards.append(self.total_reward)
            self.episode_collisions.append(self.collision_count)
            self.episode_steps.append(self.step_count)

        return self._get_obs(), reward, done, info

    def _manhattan_dist(self) -> int:
        return abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)

    def _check_collision(self) -> bool:
        for obs in self.obstacles:
            dist = np.hypot(self.agent_x - obs.x, self.agent_y - obs.y)
            if dist < obs.radius + 0.5:
                return True
        return False

    def _compute_risk_map(self) -> np.ndarray:
        """
        Build a spatial risk map by accumulating predicted obstacle positions.
        Each cell value = max probability of obstacle being there in risk_horizon steps.
        """
        risk = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        decay = 0.85  # future predictions less certain

        for obs in self.obstacles:
            predictions = obs.predict_position(self.risk_horizon, self.grid_size)
            for t, (px, py) in enumerate(predictions):
                weight = (decay ** t)
                # Gaussian spread around predicted position
                for gx in range(self.grid_size):
                    for gy in range(self.grid_size):
                        d = np.hypot(gx - px, gy - py)
                        risk[gx, gy] = max(risk[gx, gy], weight * np.exp(-0.5 * d ** 2))

        return risk

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """
        Reward shaping:
          - Reactive DQN: penalize only post-collision
          - Risk-aware extension: proximity risk penalty (proactive)
        """
        info = {"collision": False, "goal": False, "timeout": False}
        done = False
        reward = 0.0

        # Goal check
        if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
            reward += 50.0
            done = True
            info["goal"] = True
            return reward, done, info

        # Collision check (reactive)
        if self._check_collision():
            reward -= 20.0
            self.collision_count += 1
            info["collision"] = True
            done = True
            return reward, done, info

        # Step penalty (efficiency)
        reward -= 0.5

        # Progress reward
        curr_dist = self._manhattan_dist()
        reward += (self.prev_dist - curr_dist) * 1.5
        self.prev_dist = curr_dist

        # Proactive risk penalty: penalize being near predicted obstacle paths
        risk_map = self._compute_risk_map()
        ax, ay = self.agent_x, self.agent_y
        agent_risk = risk_map[ax, ay]
        reward -= agent_risk * 3.0  # proactive penalty

        # Timeout
        if self.step_count >= self.max_steps:
            reward -= 5.0
            done = True
            info["timeout"] = True

        return reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build full state observation vector."""
        gs = self.grid_size
        obs = []

        # Normalized agent position
        obs += [self.agent_x / gs, self.agent_y / gs]

        # Normalized goal relative
        obs += [(self.goal_x - self.agent_x) / gs, (self.goal_y - self.agent_y) / gs]

        # Obstacle positions and velocities
        for o in self.obstacles:
            obs += [o.x / gs, o.y / gs, o.vx, o.vy]

        # Risk map (flattened)
        risk_map = self._compute_risk_map()
        obs += risk_map.flatten().tolist()

        # Predicted future positions
        for o in self.obstacles:
            preds = o.predict_position(self.risk_horizon, gs)
            for px, py in preds:
                obs += [px / gs, py / gs]

        return np.array(obs, dtype=np.float32)

    def render_ascii(self) -> str:
        """ASCII rendering of current grid state."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        grid[self.goal_y][self.goal_x] = 'G'
        for obs in self.obstacles:
            ox, oy = int(round(obs.x)), int(round(obs.y))
            if 0 <= ox < self.grid_size and 0 <= oy < self.grid_size:
                grid[oy][ox] = 'X'
        grid[self.agent_y][self.agent_x] = 'A'

        lines = []
        lines.append('+' + '-' * self.grid_size + '+')
        for row in grid:
            lines.append('|' + ''.join(row) + '|')
        lines.append('+' + '-' * self.grid_size + '+')
        return '\n'.join(lines)

    def get_episode_stats(self) -> Dict:
        """Return rolling episode statistics."""
        if not self.episode_rewards:
            return {}
        n = len(self.episode_rewards)
        window = min(n, 50)
        return {
            "total_episodes": n,
            "avg_reward_last50": float(np.mean(self.episode_rewards[-window:])),
            "avg_collisions_last50": float(np.mean(self.episode_collisions[-window:])),
            "avg_steps_last50": float(np.mean(self.episode_steps[-window:])),
            "success_rate": float(
                sum(1 for r in self.episode_rewards[-window:] if r > 40) / window
            ),
        }
