"""
Evaluation Utilities
Provides post-training evaluation, policy rollout analysis,
and safety metric computation.
"""

import numpy as np
from typing import Dict, List, Tuple
import json


class PolicyEvaluator:
    """
    Evaluates trained policies over multiple rollouts.
    Computes safety metrics, efficiency, and robustness.
    """

    def __init__(self, env, agent, n_eval_episodes: int = 100):
        self.env = env
        self.agent = agent
        self.n_eval = n_eval_episodes

    def evaluate(self, deterministic: bool = True) -> Dict:
        """Run full evaluation sweep."""
        results = {
            "rewards": [],
            "steps": [],
            "collisions": [],
            "goals": [],
            "timeouts": [],
            "risk_exposures": [],
        }

        saved_eps = self.agent.epsilon_end if hasattr(self.agent, 'epsilon_end') else 0
        if deterministic:
            # Temporarily set epsilon to 0 for greedy evaluation
            pass

        for ep in range(self.n_eval):
            state = self.env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_collisions = 0
            ep_risk = []
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_steps += 1
                ep_risk.append(float(state[:self.env.grid_size**2].mean()) if len(state) > 10 else 0.0)

                if info.get("collision"):
                    ep_collisions += 1

                state = next_state

            results["rewards"].append(ep_reward)
            results["steps"].append(ep_steps)
            results["collisions"].append(ep_collisions)
            results["goals"].append(int(info.get("goal", False)))
            results["timeouts"].append(int(info.get("timeout", False)))
            results["risk_exposures"].append(float(np.mean(ep_risk)))

        return self._compute_summary(results)

    def _compute_summary(self, results: Dict) -> Dict:
        r = results
        n = len(r["rewards"])
        return {
            "n_episodes": n,
            "avg_reward": float(np.mean(r["rewards"])),
            "std_reward": float(np.std(r["rewards"])),
            "min_reward": float(np.min(r["rewards"])),
            "max_reward": float(np.max(r["rewards"])),
            "success_rate": float(np.mean(r["goals"])),
            "collision_rate": float(np.mean(r["collisions"])),
            "timeout_rate": float(np.mean(r["timeouts"])),
            "avg_steps": float(np.mean(r["steps"])),
            "avg_risk_exposure": float(np.mean(r["risk_exposures"])),
            "collision_free_episodes": int(sum(1 for c in r["collisions"] if c == 0)),
        }

    def compare_agents(self, other_agent, other_env) -> Dict:
        """Side-by-side comparison with another agent."""
        self_results = self.evaluate()

        orig_agent = self.agent
        orig_env = self.env
        self.agent = other_agent
        self.env = other_env
        other_results = self.evaluate()
        self.agent = orig_agent
        self.env = orig_env

        return {
            "agent_1": self_results,
            "agent_2": other_results,
            "improvements": {
                "reward": other_results["avg_reward"] - self_results["avg_reward"],
                "collision_reduction": self_results["collision_rate"] - other_results["collision_rate"],
                "success_gain": other_results["success_rate"] - self_results["success_rate"],
            }
        }


class RobustnessTest:
    """
    Tests policy robustness to distribution shifts:
      - More obstacles
      - Higher obstacle speeds
      - Smaller grid
      - Random goal positions
    """

    SCENARIOS = {
        "standard": {"num_obstacles": 5, "max_speed": 0.8, "grid_size": 12},
        "crowded":  {"num_obstacles": 8, "max_speed": 0.8, "grid_size": 12},
        "fast":     {"num_obstacles": 5, "max_speed": 1.5, "grid_size": 12},
        "small_grid": {"num_obstacles": 4, "max_speed": 0.6, "grid_size": 8},
        "stress":   {"num_obstacles": 10, "max_speed": 1.2, "grid_size": 12},
    }

    @staticmethod
    def run_scenario(env_cls, agent, scenario_name: str, n_eps: int = 50) -> Dict:
        """Evaluate agent on a specific stress scenario."""
        params = RobustnessTest.SCENARIOS.get(scenario_name, {})
        env = env_cls(**params)
        evaluator = PolicyEvaluator(env, agent, n_eps)
        results = evaluator.evaluate()
        results["scenario"] = scenario_name
        results["params"] = params
        return results


def print_comparison_table(std_results: Dict, risk_results: Dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "="*65)
    print(f"{'Metric':<30} {'Standard DQN':>15} {'Risk-Aware DQN':>15}")
    print("-"*65)

    metrics = [
        ("Avg Reward", "avg_reward", ".2f"),
        ("Std Reward", "std_reward", ".2f"),
        ("Success Rate", "success_rate", ".1%"),
        ("Collision Rate (per ep)", "collision_rate", ".3f"),
        ("Timeout Rate", "timeout_rate", ".1%"),
        ("Avg Steps", "avg_steps", ".1f"),
        ("Avg Risk Exposure", "avg_risk_exposure", ".3f"),
        ("Collision-free Episodes", "collision_free_episodes", "d"),
    ]

    for label, key, fmt in metrics:
        s = std_results.get(key, 0)
        r = risk_results.get(key, 0)
        better = "✓" if (
            (key in ("avg_reward", "success_rate", "collision_free_episodes") and r > s) or
            (key in ("collision_rate", "timeout_rate", "avg_risk_exposure") and r < s)
        ) else ""
        print(f"  {label:<28} {s:{fmt}:>15} {r:{fmt}:>14} {better}")

    print("="*65)
    improvement = ((risk_results.get("avg_reward",0) - std_results.get("avg_reward",0)) /
                   max(abs(std_results.get("avg_reward",1)), 1)) * 100
    print(f"\n  Overall reward improvement: {improvement:+.1f}%")
    coll_red = std_results.get("collision_rate",0) - risk_results.get("collision_rate",0)
    print(f"  Collision reduction: {coll_red:.3f} per episode")
    print()


if __name__ == "__main__":
    print("Evaluation utilities ready.")
    print("Usage:")
    print("  from evaluate import PolicyEvaluator, print_comparison_table")
    print("  evaluator = PolicyEvaluator(env, agent)")
    print("  results = evaluator.evaluate()")
