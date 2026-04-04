"""
optimizer/gp_optimizer.py

Gaussian Process Bayesian Optimization over the discrete prompt search space.

Why GP-BO?
----------
We have a small evaluation budget (each "function evaluation" requires running
N real conversations, which costs time and money). GP-BO is optimal in this
regime because:

1. It builds a probabilistic surrogate model (Gaussian Process) of the reward
   function over the search space.
2. It uses Upper Confidence Bound (UCB) acquisition to balance exploration
   (try uncertain regions) vs exploitation (try promising regions).
3. It handles noisy observations naturally — each batch of calls gives a
   noisy estimate of the true reward for that config.
4. It finds good solutions in far fewer evaluations than grid search or
   random search.

Search space:
  4 axes × {2–3 values each} = 54 total configurations.
  We represent each config as a flat integer vector [a0, a1, a2, a3].
  scikit-optimize treats each dimension as an Integer space.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from agent.config import PROMPT_AXES, AXIS_SIZES


class GPBayesianOptimizer:
    def __init__(self, n_random_starts: int = 3, noise: float = 0.05):
        """
        Args:
            n_random_starts: Number of random evaluations before GP kicks in.
                             Should be >= n_dimensions to seed the GP well.
            noise: Expected noise level in reward estimates (std of call-to-call
                   variance). Tells the GP not to overfit single noisy readings.
        """
        self.n_random_starts = n_random_starts
        self.noise = noise

        # Build the search space: one Integer dimension per prompt axis
        self.axis_names = list(AXIS_SIZES.keys())
        self.space = [
            Integer(0, size - 1, name=axis_name)
            for axis_name, size in AXIS_SIZES.items()
        ]

        # History: list of (config_dict, reward) tuples
        self.history: List[Tuple[Dict[str, int], float]] = []

        # Raw skopt result (populated after optimize() call)
        self._skopt_result = None

    def vector_to_config(self, vec: List[int]) -> Dict[str, int]:
        """Convert integer vector to axis config dict."""
        return {name: val for name, val in zip(self.axis_names, vec)}

    def config_to_vector(self, config: Dict[str, int]) -> List[int]:
        """Convert axis config dict to integer vector."""
        return [config[name] for name in self.axis_names]

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, int]], float],
        n_iterations: int = 6,
    ) -> Dict[str, int]:
        """
        Run the GP-BO loop.

        Args:
            objective_fn: Function that takes a config dict and returns a reward scalar.
                          This wraps the full simulation + scoring pipeline.
            n_iterations: Total number of configs to evaluate (including random starts).

        Returns:
            Best config dict found.
        """
        call_count = [0]

        def skopt_objective(vec):
            """skopt minimizes, so we negate the reward."""
            config = self.vector_to_config(vec)
            reward = objective_fn(config)
            self.history.append((config, reward))
            call_count[0] += 1
            return -reward  # negate for minimization

        self._skopt_result = gp_minimize(
            func=skopt_objective,
            dimensions=self.space,
            n_calls=n_iterations,
            n_random_starts=self.n_random_starts,
            acq_func="LCB",         # Lower Confidence Bound (skopt minimizes, so LCB = UCB semantically)
            kappa=1.96,             # exploration–exploitation tradeoff (≈95% CI)
            noise=self.noise,
            random_state=42,
            verbose=False,
        )

        best_vec = self._skopt_result.x
        return self.vector_to_config(best_vec)

    def best_config(self) -> Optional[Dict[str, int]]:
        """Return the best config seen so far."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x[1])[0]

    def best_reward(self) -> float:
        """Return the best reward seen so far."""
        if not self.history:
            return 0.0
        return max(r for _, r in self.history)

    def convergence_data(self) -> Dict:
        """Return data for plotting the optimization curve."""
        rewards = [r for _, r in self.history]
        running_best = [max(rewards[: i + 1]) for i in range(len(rewards))]
        return {
            "iterations": list(range(1, len(rewards) + 1)),
            "rewards": rewards,
            "running_best": running_best,
        }

    def save_history(self, path: str) -> None:
        """Persist optimization history to JSON."""
        def to_serializable(obj):
            """Convert numpy types to native Python for JSON serialization."""
            if hasattr(obj, "item"):  # numpy scalar (int64, float32, etc.)
                return obj.item()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        data = {
            "history": [
                {"iteration": i + 1, "config": to_serializable(cfg), "reward": float(r)}
                for i, (cfg, r) in enumerate(self.history)
            ],
            "best_config": to_serializable(self.best_config()),
            "best_reward": float(self.best_reward()),
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_history(self, path: str) -> None:
        """Load a previously saved history (for resuming runs)."""
        with open(path) as f:
            data = json.load(f)
        self.history = [
            (entry["config"], entry["reward"]) for entry in data["history"]
        ]
