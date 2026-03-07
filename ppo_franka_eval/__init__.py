"""
ppo_franka_eval — PPO evaluation utilities for real Franka FR3.

Provides a self-contained, real-time-safe PPO actor and evaluator
designed for deployment on the physical robot.  All inference is
CPU-only with pre-allocated tensors to avoid violating the 1 ms
libfranka control budget.

Key classes
-----------
PPOActorInference      — loads an actor checkpoint, warms up BLAS/kernel
                         caches, and provides deterministic mean-action
                         inference via a pre-allocated input buffer.
WelfordNormalizerInference
                       — read-only normalizer loaded from .npz checkpoint.
PPOFrankaEvaluator     — builds the 25-dim observation from real robot state,
                         normalises it, and runs the actor at 50 Hz
                         (every action_repeat=20 control cycles).
"""

from .ppo_actor import PPOActorInference
from .ppo_franka_evaluator import PPOFrankaEvaluator, WelfordNormalizerInference

__all__ = [
    "PPOActorInference",
    "PPOFrankaEvaluator",
    "WelfordNormalizerInference",
]
