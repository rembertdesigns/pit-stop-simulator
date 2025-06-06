import numpy as np
import matplotlib.pyplot as plt

# ✅ Load total reward trends (manually saved from training)
q_rewards = np.load("data/q_learning_rewards.npy")
ppo_rewards = np.load("data/ppo_rewards.npy")

# ✅ Plot reward trends over episodes
plt.figure(figsize=(12, 6))
plt.plot(q_rewards, label="Q-learning", alpha=0.8)
plt.plot(ppo_rewards, label="PPO", alpha=0.8)
plt.title("Total Episode Reward Over Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
