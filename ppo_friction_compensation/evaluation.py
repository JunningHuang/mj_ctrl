import numpy as np
import matplotlib.pyplot as plt
from ppo_friction_compensation.ppo_agent import PPOAgent
from ppo_friction_compensation.env_wrapper import HybridControlEnv, WelfordNormalizer

CHECKPOINT = "/tmp/ppo_test/final"  # or epoch_0003

# Load agent
agent = PPOAgent(obs_dim=18, act_dim=7)
agent.load(CHECKPOINT)
agent.actor.eval()  # disable dropout etc. (none here, but good practice)

# Load env with the same normalizer stats from training
env = HybridControlEnv()
env.normalizer.load(f"{CHECKPOINT}_normalizer.npz")

# Run one episode
obs = env.reset()
force_errors, delta_taus, rewards = [], [], []

for step in range(600):
    act, _, _ = agent.step(obs)      # stochastic — samples from policy
    obs, rew, done, info = env.step(act)

    # obs_raw isn't directly accessible post-normalize, so reconstruct force_error
    force_errors.append(obs[0] * env.normalizer.std[0] + env.normalizer.mean[0])
    delta_taus.append(act.copy())
    rewards.append(rew)

    if done:
        print(f"Done at step {step}: {info['termination']}")
        break

print(f"Total return: {sum(rewards):.2f}")
print(f"Mean |force_error|: {np.mean(np.abs(force_errors)):.3f} N")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(force_errors); axes[0].axhline(0, color='r', linestyle='--')
axes[0].set_ylabel("Force error [N]"); axes[0].set_title("Force Error (0 = perfect)")
axes[1].plot(delta_taus);  axes[1].set_ylabel("Δτ [Nm]"); axes[1].set_title("PPO torque corrections")
axes[2].plot(np.cumsum(rewards)); axes[2].set_ylabel("Cumulative return"); axes[2].set_xlabel("Step")
plt.tight_layout()
plt.savefig("/tmp/ppo_eval.png")
plt.show()