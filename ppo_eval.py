import numpy as np
from stable_baselines3 import PPO
from env.gym_race_env import PitStopEnv

# ✅ Load trained PPO model
model_path = "models/ppo_pit_stop"
model = PPO.load(model_path)

# ✅ Create environment
env = PitStopEnv()

# 🏁 Start evaluation
obs, _ = env.reset()
done = False
total_reward = 0
pit_laps = []

print("\n🏁 Starting evaluation...")

while not done:
    action, _ = model.predict(obs)
    if action == 1:
        pit_laps.append(int(obs[0]))  # Track the lap number when pitting

    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    action_str = "PIT" if action == 1 else "STAY OUT"
    print(f"Lap {int(obs[0])} | Action: {action_str} | Tire Wear: {obs[1]:.2f} | Traffic: {obs[2]:.2f} | Reward: {reward:.2f}")

# ✅ Save pit laps for later comparison
import os
os.makedirs("data", exist_ok=True)
np.save("data/ppo_pit_decisions.npy", [pit_laps])

print("\n✅ Evaluation complete.")
print(f"Total reward: {total_reward:.2f}")
print(f"Pit stops occurred at laps: {pit_laps}")
