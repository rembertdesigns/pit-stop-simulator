from stable_baselines3 import PPO
from env.gym_race_env import PitStopEnv
import numpy as np
import os

print("🏁 Starting evaluation...")

env = PitStopEnv()
model = PPO.load("models/ppo_pit_stop")

total_episodes = 1
ppo_rewards = []
pit_decisions = []

for episode in range(total_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    episode_pits = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if action == 1:
            episode_pits.append(int(obs[0]))  # Lap number after pit

        total_reward += reward

        print(f"Lap {int(obs[0])} | Action: {'PIT' if action == 1 else 'STAY OUT'} | "
              f"Tire Wear: {env.tire_wear:.2f} | Traffic: {env.traffic:.2f} | Reward: {reward:.2f}")

    ppo_rewards.append(total_reward)
    pit_decisions.append(episode_pits)

print("\n✅ Evaluation complete.")
print(f"Total reward: {ppo_rewards[-1]:.2f}")
print(f"Pit stops occurred at laps: {pit_decisions[-1]}")

# ✅ Save rewards for comparison
os.makedirs("data", exist_ok=True)
np.save("data/ppo_rewards.npy", ppo_rewards)

