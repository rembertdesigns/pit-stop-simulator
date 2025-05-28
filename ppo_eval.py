import numpy as np
from stable_baselines3 import PPO
from env.gym_race_env import PitStopEnv
import os

# ✅ Load trained PPO model
model_path = "models/ppo_pit_stop"
model = PPO.load(model_path)

# ✅ Create environment
env = PitStopEnv()

# 🔁 Run multiple episodes
all_pit_decisions = []
num_episodes = 100

print("\n🏁 Starting PPO Evaluation...")

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    pit_laps = []

    while not done:
        action, _ = model.predict(obs)
        if action == 1:
            pit_laps.append(int(obs[0]))
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    all_pit_decisions.append(pit_laps)

    if (ep + 1) % 10 == 0:
        print(f"✅ Episode {ep + 1} complete | Pit stops: {pit_laps}")

# ✅ Save PPO pit data
os.makedirs("data", exist_ok=True)
np.save("data/ppo_pit_decisions.npy", np.array(all_pit_decisions, dtype=object))

print("\n✅ Saved: data/ppo_pit_decisions.npy")

