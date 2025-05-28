from stable_baselines3 import PPO
from env.gym_race_env import PitStopEnv

# Load environment and trained model
env = PitStopEnv()
model = PPO.load("models/ppo_pit_stop")

# Reset environment
obs, _ = env.reset()
done = False
total_reward = 0
pit_laps = []

print("🏁 Starting evaluation...")

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    lap = int(obs[0])
    total_reward += reward

    print(f"Lap {lap} | Action: {'PIT' if action == 1 else 'STAY OUT'} | Tire Wear: {obs[1]:.2f} | Traffic: {obs[2]:.2f} | Reward: {reward:.2f}")

    if action == 1:
        pit_laps.append(lap)

    done = terminated or truncated

print("\n✅ Evaluation complete.")
print(f"Total reward: {total_reward:.2f}")
print(f"Pit stops occurred at laps: {pit_laps}")
