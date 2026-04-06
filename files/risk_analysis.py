"""
Risk Analysis Utilities
Provides tools for:
  - Spatial risk map generation
  - Obstacle trajectory prediction (linear + Kalman-like)
  - Safety margin computation
  - Risk-weighted action evaluation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class TrajectoryPredictor:
    """
    Predicts future obstacle positions using:
      1. Linear velocity extrapolation (baseline)
      2. Exponential weighted moving average on velocity (adaptive)
    """

    def __init__(self, grid_size: int, horizon: int = 8):
        self.grid_size = grid_size
        self.horizon = horizon
        self.history: Dict[int, List[Tuple[float, float]]] = {}
        self.velocities: Dict[int, Tuple[float, float]] = {}

    def update(self, obstacle_id: int, x: float, y: float) -> None:
        """Update history for obstacle."""
        if obstacle_id not in self.history:
            self.history[obstacle_id] = []
        self.history[obstacle_id].append((x, y))
        if len(self.history[obstacle_id]) > 10:
            self.history[obstacle_id].pop(0)

        # Estimate velocity from recent history
        hist = self.history[obstacle_id]
        if len(hist) >= 2:
            vx = hist[-1][0] - hist[-2][0]
            vy = hist[-1][1] - hist[-2][1]
            # EWMA smoothing
            if obstacle_id in self.velocities:
                pvx, pvy = self.velocities[obstacle_id]
                alpha = 0.4
                vx = alpha * vx + (1 - alpha) * pvx
                vy = alpha * vy + (1 - alpha) * pvy
            self.velocities[obstacle_id] = (vx, vy)

    def predict(
        self, obstacle_id: int, current_x: float, current_y: float
    ) -> List[Tuple[float, float]]:
        """Predict future positions with wall bounce."""
        if obstacle_id not in self.velocities:
            return [(current_x, current_y)] * self.horizon

        vx, vy = self.velocities[obstacle_id]
        positions = []
        px, py = current_x, current_y
        pvx, pvy = vx, vy

        for _ in range(self.horizon):
            px += pvx
            py += pvy
            gs = self.grid_size
            if px <= 0 or px >= gs - 1:
                pvx *= -1
                px = np.clip(px, 0, gs - 1)
            if py <= 0 or py >= gs - 1:
                pvy *= -1
                py = np.clip(py, 0, gs - 1)
            positions.append((float(px), float(py)))

        return positions


class RiskMapper:
    """
    Generates multi-step spatial risk maps for proactive navigation.
    Combines current obstacle proximity + predicted trajectory risk.
    """

    def __init__(
        self,
        grid_size: int,
        risk_horizon: int = 6,
        sigma: float = 1.2,
        decay: float = 0.8
    ):
        self.grid_size = grid_size
        self.risk_horizon = risk_horizon
        self.sigma = sigma
        self.decay = decay

        # Precompute grid coordinates
        xs = np.arange(grid_size)
        ys = np.arange(grid_size)
        self.GX, self.GY = np.meshgrid(xs, ys)

    def gaussian_risk(self, cx: float, cy: float, weight: float = 1.0) -> np.ndarray:
        """Gaussian risk blob centered at (cx, cy)."""
        d2 = (self.GX - cx) ** 2 + (self.GY - cy) ** 2
        return weight * np.exp(-d2 / (2 * self.sigma ** 2))

    def compute(
        self,
        obstacles: List[Dict],  # [{"x": float, "y": float, "vx": float, "vy": float}]
        agent_x: Optional[float] = None,
        agent_y: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute full risk map.
        Returns: (grid_size, grid_size) array in [0, 1].
        """
        risk = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for obs in obstacles:
            ox, oy = obs["x"], obs["y"]
            ovx, ovy = obs.get("vx", 0), obs.get("vy", 0)

            # Current position risk (highest weight)
            risk += self.gaussian_risk(ox, oy, weight=1.0)

            # Future position risks (decaying weight)
            px, py = ox, oy
            pvx, pvy = ovx, ovy
            gs = self.grid_size
            for t in range(1, self.risk_horizon + 1):
                px += pvx
                py += pvy
                if px <= 0 or px >= gs - 1:
                    pvx *= -1
                    px = np.clip(px, 0, gs - 1)
                if py <= 0 or py >= gs - 1:
                    pvy *= -1
                    py = np.clip(py, 0, gs - 1)
                weight = self.decay ** t
                risk += self.gaussian_risk(px, py, weight=weight)

        # Normalize to [0, 1]
        max_r = risk.max()
        if max_r > 0:
            risk = risk / max_r

        return risk.T  # transpose for (x, y) indexing

    def compute_agent_risk(
        self, risk_map: np.ndarray, ax: int, ay: int
    ) -> float:
        """Risk at agent's current position."""
        if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
            return float(risk_map[ax, ay])
        return 0.0

    def safe_actions(
        self,
        risk_map: np.ndarray,
        agent_x: int,
        agent_y: int,
        threshold: float = 0.5
    ) -> List[int]:
        """Return list of action indices that lead to below-threshold risk cells."""
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # U D L R Stay
        safe = []
        gs = self.grid_size
        for i, (dx, dy) in enumerate(moves):
            nx = int(np.clip(agent_x + dx, 0, gs - 1))
            ny = int(np.clip(agent_y + dy, 0, gs - 1))
            if risk_map[nx, ny] < threshold:
                safe.append(i)
        return safe if safe else list(range(5))


class SafetyMetrics:
    """Computes safety and efficiency metrics for evaluation."""

    @staticmethod
    def collision_rate(collisions: List[int], window: int = 50) -> List[float]:
        rates = []
        for i in range(len(collisions)):
            w = max(0, i - window)
            rates.append(float(np.mean(collisions[w:i+1])))
        return rates

    @staticmethod
    def time_to_safety(risk_values: List[float], threshold: float = 0.3) -> float:
        """Average steps spent in high-risk zones."""
        high_risk = sum(1 for r in risk_values if r > threshold)
        return high_risk / len(risk_values) if risk_values else 0.0

    @staticmethod
    def path_efficiency(steps_taken: int, optimal_steps: int) -> float:
        """Ratio of optimal to actual steps (1.0 = optimal)."""
        return optimal_steps / max(steps_taken, 1)

    @staticmethod
    def risk_adjusted_reward(reward: float, collision_penalty: float,
                              risk_exposure: float, weight: float = 0.3) -> float:
        """Reward adjusted for risk exposure."""
        return reward - weight * risk_exposure * collision_penalty

    @staticmethod
    def compute_improvement(std_metric: float, risk_metric: float) -> Dict:
        """Compute percentage improvement of risk-aware over standard."""
        if std_metric == 0:
            return {"absolute": 0, "relative_pct": 0}
        diff = risk_metric - std_metric
        pct = (diff / abs(std_metric)) * 100
        return {"absolute": round(diff, 3), "relative_pct": round(pct, 1)}


def generate_risk_map_demo(
    grid_size: int = 12,
    n_obstacles: int = 4,
    step: int = 0,
    seed: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate a demo risk map with animated obstacles (for UI demo).
    Returns: (risk_map, obstacles_list)
    """
    rng = np.random.default_rng(seed)
    obstacles = []

    for i in range(n_obstacles):
        base_x = rng.uniform(2, grid_size - 3)
        base_y = rng.uniform(2, grid_size - 3)
        speed = rng.uniform(0.4, 0.9)
        angle_base = rng.uniform(0, 2 * np.pi)

        # Animate position over steps
        t = step * 0.05
        angle = angle_base + t
        ox = base_x + 2.5 * np.cos(angle + i * np.pi / 2)
        oy = base_y + 2.5 * np.sin(angle + i * np.pi / 2)
        ox = float(np.clip(ox, 1, grid_size - 2))
        oy = float(np.clip(oy, 1, grid_size - 2))

        vx = speed * np.cos(angle + np.pi / 2)
        vy = speed * np.sin(angle + np.pi / 2)

        obstacles.append({"x": ox, "y": oy, "vx": float(vx), "vy": float(vy)})

    mapper = RiskMapper(grid_size)
    risk_map = mapper.compute(obstacles)
    return risk_map, obstacles
