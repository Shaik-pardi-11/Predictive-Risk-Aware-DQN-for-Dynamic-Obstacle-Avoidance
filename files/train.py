"""
Training & Experiment Runner
Trains both Standard DQN and Risk-Aware DQN side-by-side.
Tracks metrics for comparison visualization.
"""

import numpy as np
import random
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from environment import DynamicNavEnvironment
from agents import StandardDQNAgent, RiskAwareDQNAgent


@dataclass
class EpisodeLog:
    episode: int
    agent_type: str
    total_reward: float
    steps: int
    collisions: int
    goal_reached: bool
    avg_loss: float
    epsilon: float
    q_value: float


class ExperimentRunner:
    """
    Runs training experiments comparing Standard DQN vs Risk-Aware DQN.
    Outputs structured logs for visualization.
    """

    def __init__(
        self,
        grid_size: int = 12,
        num_obstacles: int = 5,
        max_steps: int = 150,
        risk_horizon: int = 5,
        n_episodes: int = 500,
        seed: int = 42
    ):
        self.n_episodes = n_episodes
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Create environments
        self.env_std = DynamicNavEnvironment(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            max_steps=max_steps,
            risk_horizon=risk_horizon,
            seed=seed
        )
        self.env_risk = DynamicNavEnvironment(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            max_steps=max_steps,
            risk_horizon=risk_horizon,
            seed=seed
        )

        state_dim = self.env_std.obs_dim
        action_dim = self.env_std.action_space_n

        # Create agents
        agent_kwargs = dict(
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=8000,
            batch_size=64,
            target_update_freq=100,
            buffer_capacity=30000,
            hidden_dims=[256, 128],
        )

        self.std_agent = StandardDQNAgent(state_dim, action_dim, **agent_kwargs)
        self.risk_agent = RiskAwareDQNAgent(state_dim, action_dim, risk_weight=0.25, **agent_kwargs)

        self.logs_std: List[EpisodeLog] = []
        self.logs_risk: List[EpisodeLog] = []

    def _run_episode(self, env, agent, is_risk_aware: bool) -> EpisodeLog:
        state = env.reset()
        total_reward = 0.0
        collisions = 0
        done = False
        ep_loss = []
        ep_q = []

        while not done:
            if is_risk_aware:
                action = agent.select_action(state)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            agent.push_experience(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)

            if info.get("collision"):
                collisions += 1
            total_reward += reward
            state = next_state

        metrics = agent.get_metrics_summary()
        return EpisodeLog(
            episode=len(self.logs_std if not is_risk_aware else self.logs_risk),
            agent_type="risk_aware" if is_risk_aware else "standard",
            total_reward=round(total_reward, 3),
            steps=env.step_count,
            collisions=collisions,
            goal_reached=info.get("goal", False),
            avg_loss=round(float(np.mean(ep_loss)) if ep_loss else 0.0, 4),
            epsilon=metrics["epsilon"],
            q_value=round(metrics["avg_q"], 4),
        )

    def run(self, verbose: bool = True) -> Dict:
        """Train both agents and return comparison results."""
        print(f"Starting training: {self.n_episodes} episodes each")
        print(f"State dim: {self.env_std.obs_dim}, Actions: {self.env_std.action_space_n}")
        print("=" * 60)

        start_time = time.time()

        for ep in range(self.n_episodes):
            # Train standard DQN
            log_std = self._run_episode(self.env_std, self.std_agent, is_risk_aware=False)
            self.logs_std.append(log_std)

            # Train risk-aware DQN
            log_risk = self._run_episode(self.env_risk, self.risk_agent, is_risk_aware=True)
            self.logs_risk.append(log_risk)

            if verbose and ep % 50 == 0:
                self._print_progress(ep, log_std, log_risk)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")
        return self._compile_results()

    def _print_progress(self, ep: int, std: EpisodeLog, risk: EpisodeLog) -> None:
        w = min(ep + 1, 50)
        std_rewards = [l.total_reward for l in self.logs_std[-w:]]
        risk_rewards = [l.total_reward for l in self.logs_risk[-w:]]
        std_coll = [l.collisions for l in self.logs_std[-w:]]
        risk_coll = [l.collisions for l in self.logs_risk[-w:]]

        print(f"Ep {ep:4d} | "
              f"Std: R={np.mean(std_rewards):6.1f} Coll={np.mean(std_coll):.2f} ε={std.epsilon:.3f} | "
              f"Risk: R={np.mean(risk_rewards):6.1f} Coll={np.mean(risk_coll):.2f} ε={risk.epsilon:.3f}")

    def _compile_results(self) -> Dict:
        def smooth(values, w=20):
            out = []
            for i, v in enumerate(values):
                start = max(0, i - w)
                out.append(float(np.mean(values[start:i+1])))
            return out

        std_rewards = [l.total_reward for l in self.logs_std]
        risk_rewards = [l.total_reward for l in self.logs_risk]
        std_colls = [l.collisions for l in self.logs_std]
        risk_colls = [l.collisions for l in self.logs_risk]
        std_goals = [int(l.goal_reached) for l in self.logs_std]
        risk_goals = [int(l.goal_reached) for l in self.logs_risk]
        std_steps = [l.steps for l in self.logs_std]
        risk_steps = [l.steps for l in self.logs_risk]

        return {
            "config": {
                "n_episodes": self.n_episodes,
                "grid_size": self.env_std.grid_size,
                "num_obstacles": self.env_std.num_obstacles,
            },
            "standard": {
                "rewards": std_rewards,
                "rewards_smooth": smooth(std_rewards),
                "collisions": std_colls,
                "collisions_smooth": smooth(std_colls),
                "goal_reached": std_goals,
                "steps": std_steps,
                "final_avg_reward": float(np.mean(std_rewards[-50:])),
                "final_collision_rate": float(np.mean(std_colls[-50:])),
                "final_success_rate": float(np.mean(std_goals[-50:])),
                "total_collisions": sum(std_colls),
            },
            "risk_aware": {
                "rewards": risk_rewards,
                "rewards_smooth": smooth(risk_rewards),
                "collisions": risk_colls,
                "collisions_smooth": smooth(risk_colls),
                "goal_reached": risk_goals,
                "steps": risk_steps,
                "final_avg_reward": float(np.mean(risk_rewards[-50:])),
                "final_collision_rate": float(np.mean(risk_colls[-50:])),
                "final_success_rate": float(np.mean(risk_goals[-50:])),
                "total_collisions": sum(risk_colls),
            },
        }

    def save_results(self, path: str = "results.json") -> None:
        results = self._compile_results()
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")


def simulate_training_data(n_episodes: int = 300, seed: int = 42) -> Dict:
    """
    Generate realistic simulated training data for visualization
    without running full training (fast path for demo).
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    episodes = list(range(n_episodes))

    def learning_curve(
        start: float, end: float, noise: float,
        collision_start: float, collision_end: float,
        n: int, delay: int = 30, faster: bool = False
    ):
        rewards, collisions = [], []
        decay = 0.012 if faster else 0.008
        for i in range(n):
            progress = 1 - np.exp(-decay * max(0, i - delay))
            r = start + (end - start) * progress
            r += rng.normal(0, noise * (1 - 0.5 * progress))
            rewards.append(float(r))

            c_progress = 1 - np.exp(-decay * max(0, i - delay))
            c = collision_start + (collision_end - collision_start) * c_progress
            c += abs(rng.normal(0, 0.3 * (1 - 0.5 * c_progress)))
            collisions.append(max(0.0, float(c)))

        return rewards, collisions

    # Standard DQN: slower learning, more collisions
    std_r, std_c = learning_curve(
        start=-25, end=18, noise=12,
        collision_start=2.8, collision_end=0.9,
        n=n_episodes, delay=40, faster=False
    )

    # Risk-Aware DQN: faster convergence, fewer collisions
    risk_r, risk_c = learning_curve(
        start=-20, end=32, noise=8,
        collision_start=2.2, collision_end=0.2,
        n=n_episodes, delay=20, faster=True
    )

    def smooth(v, w=15):
        return [float(np.mean(v[max(0,i-w):i+1])) for i in range(len(v))]

    def goal_rate(rewards, threshold=15.0):
        rates = []
        for i, r in enumerate(rewards):
            w = max(0, i-20)
            rates.append(float(np.mean([1.0 if x > threshold else 0.0 for x in rewards[w:i+1]])))
        return rates

    return {
        "config": {"n_episodes": n_episodes, "grid_size": 12, "num_obstacles": 5},
        "episodes": episodes,
        "standard": {
            "rewards": std_r,
            "rewards_smooth": smooth(std_r),
            "collisions": std_c,
            "collisions_smooth": smooth(std_c),
            "goal_rate": goal_rate(std_r),
            "final_avg_reward": float(np.mean(std_r[-50:])),
            "final_collision_rate": float(np.mean(std_c[-50:])),
            "final_success_rate": float(np.mean([1 if r > 15 else 0 for r in std_r[-50:]])),
            "total_collisions": int(sum(std_c)),
        },
        "risk_aware": {
            "rewards": risk_r,
            "rewards_smooth": smooth(risk_r),
            "collisions": risk_c,
            "collisions_smooth": smooth(risk_c),
            "goal_rate": goal_rate(risk_r),
            "final_avg_reward": float(np.mean(risk_r[-50:])),
            "final_collision_rate": float(np.mean(risk_c[-50:])),
            "final_success_rate": float(np.mean([1 if r > 15 else 0 for r in risk_r[-50:]])),
            "total_collisions": int(sum(risk_c)),
        },
    }


if __name__ == "__main__":
    # Quick training run
    runner = ExperimentRunner(n_episodes=200)
    results = runner.run(verbose=True)
    runner.save_results("results.json")

    print("\n=== Final Comparison ===")
    print(f"Standard DQN   | Reward: {results['standard']['final_avg_reward']:.1f} "
          f"| Collisions: {results['standard']['final_collision_rate']:.2f} "
          f"| Success: {results['standard']['final_success_rate']*100:.1f}%")
    print(f"Risk-Aware DQN | Reward: {results['risk_aware']['final_avg_reward']:.1f} "
          f"| Collisions: {results['risk_aware']['final_collision_rate']:.2f} "
          f"| Success: {results['risk_aware']['final_success_rate']*100:.1f}%")
