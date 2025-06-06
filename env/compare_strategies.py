import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load pit decision data
q_learning_pits = np.load("data/q_learning_pit_decisions.npy", allow_pickle=True)
ppo_pits = np.load("data/ppo_pit_decisions.npy", allow_pickle=True)

def plot_pit_comparison(q_pits, ppo_pits, total_laps=58, bins=10):
    def build_heatmap(pit_data):
        heatmap = np.zeros((bins, total_laps))
        episodes_per_bin = len(pit_data) // bins
        for b in range(bins):
            start = b * episodes_per_bin
            end = (b + 1) * episodes_per_bin
            for episode in pit_data[start:end]:
                for lap in episode:
                    if lap < total_laps:
                        heatmap[b, lap] += 1
        return heatmap

    q_heatmap = build_heatmap(q_pits)
    ppo_heatmap = build_heatmap(ppo_pits)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    sns.heatmap(q_heatmap, ax=axes[0], cmap="Blues", xticklabels=5)
    axes[0].set_title("Q-Learning Pit Strategy Over Time")
    axes[0].set_ylabel("Episode Bin")
    axes[0].set_xlabel("Lap")

    sns.heatmap(ppo_heatmap, ax=axes[1], cmap="Greens", xticklabels=5)
    axes[1].set_title("PPO Pit Strategy Over Time")
    axes[1].set_ylabel("Episode Bin")
    axes[1].set_xlabel("Lap")

    plt.tight_layout()
    plt.show()

plot_pit_comparison(q_learning_pits, ppo_pits)
