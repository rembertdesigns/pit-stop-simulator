from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.gym_race_env import PitStopEnv
import numpy as np
import os

# ✅ Step 1: Create and validate the environment
env = PitStopEnv()
check_env(env, warn=True)  # Optional check

# ✅ Step 2: Train PPO agent
model = PPO(
    "MlpPolicy",           # Use multi-layer perceptron policy
    env,
    verbose=1,
    n_steps=2048,          # Rollout buffer size
    batch_size=64,         # Mini-batch size
    gae_lambda=0.95,       # GAE smoothing
    gamma=0.99,            # Discount factor
    ent_coef=0.01,         # Entropy coefficient
    learning_rate=2.5e-4,  # Learning rate
    n_epochs=10            # PPO epochs per update
)

# ✅ Step 3: Train the model
model.learn(total_timesteps=300_000)

# ✅ Step 4: Save the trained model
model_path = "models/ppo_pit_stop"
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# ✅ Step 5: Manually simulate one episode to collect reward
rewards = []
obs, _ = env.reset()
total_reward = 0

for _ in range(env.total_laps):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    if done:
        break

rewards.append(total_reward)

# ✅ Step 6: Save reward data for comparison
os.makedirs("data", exist_ok=True)
np.save("data/ppo_rewards.npy", np.array(rewards))
print("✅ PPO reward saved to: data/ppo_rewards.npy")
