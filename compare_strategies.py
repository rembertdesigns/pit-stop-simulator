import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load pit decision data
q_learning_decisions = np.load("data/q_learning_pit_decisions.npy", allow_pickle=True)
ppo_decisions = np.load("data/ppo_pit_decisions.npy", allow_pickle=True)

# ✅ Helper: Build heatmap matrix
def build_heatmap(pit_decisions, total_laps=58, bins=10):
    heatmap = np.zeros((bins, total_laps))
    episodes_per_bin = len(pit_decisions) // bins

    for b in range(bins):
        start = b * episodes_per_bin
        end = (b + 1) * episodes_per_bin
        for episode in pit_decisions[start:end]:
            for lap in episode:
                if 0 <= lap < total_laps:
                    heatmap[b, lap] += 1
    return heatmap

# ✅ Generate heatmaps
q_heatmap = build_heatmap(q_learning_decisions)
ppo_heatmap = build_heatmap(ppo_decisions)

# ✅ Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

sns.heatmap(q_heatmap, ax=axes[0], cmap="YlGnBu", xticklabels=5, yticklabels=False)
axes[0].set_title("Q-learning Pit Stops")
axes[0].set_xlabel("Lap Number")
axes[0].set_ylabel("Training Stage")

sns.heatmap(ppo_heatmap, ax=axes[1], cmap="YlGnBu", xticklabels=5, yticklabels=False)
axes[1].set_title("PPO Pit Stops")
axes[1].set_xlabel("Lap Number")

plt.tight_layout()
plt.show()
