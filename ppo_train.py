from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.gym_race_env import PitStopEnv
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
    n_epochs=10,           # PPO epochs per update
)

# ✅ Step 3: Train for 100k timesteps (adjust if needed)
model.learn(total_timesteps=100_000)

# ✅ Step 4: Save the model
model_path = "models/ppo_pit_stop"
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"✅ Model saved to: {model_path}")
