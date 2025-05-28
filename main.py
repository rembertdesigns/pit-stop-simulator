from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train_q_learning(episodes=500):
    env = PitStopEnv()
    agent = QAgent(env)
    rewards = []
    pit_decisions = []  # Track pit laps

    for episode in range(episodes):
        obs = env.reset()
        state = agent.discretize(obs)
        total_reward = 0
        done = False
        current_pit_laps = []

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = agent.discretize(next_obs)

            # Track pit decisions by lap
            if action == 1:
                lap_num = int(next_obs[0])  # get lap after the action was applied
                current_pit_laps.append(lap_num)

            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)
        pit_decisions.append(current_pit_laps)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    return rewards, pit_decisions

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Progress")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pit_heatmap(pit_decisions, total_laps=58, bins=10):
    heatmap = np.zeros((bins, total_laps))
    episodes_per_bin = len(pit_decisions) // bins

    for b in range(bins):
        start = b * episodes_per_bin
        end = (b + 1) * episodes_per_bin
        for episode in pit_decisions[start:end]:
            for lap in episode:
                if lap < total_laps:
                    heatmap[b, lap] += 1

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap,
        cmap="YlGnBu",
        xticklabels=5,
        yticklabels=[f"Ep {i*episodes_per_bin}" for i in range(bins)]
    )
    plt.xlabel("Lap Number")
    plt.ylabel("Training Progress")
    plt.title("Pit Stop Frequency by Lap and Training Stage")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards, pit_decisions = train_q_learning()
    plot_rewards(rewards)

    print("First few pit decisions:", pit_decisions[:3])  # 👈 Add this
    plot_pit_heatmap(pit_decisions)


