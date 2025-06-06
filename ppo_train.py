import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Ensure this path correctly points to your updated PitStopEnv
from env.gym_race_env import PitStopEnv

def main():
    # --- Environment Setup ---
    # Parameters for your PitStopEnv during training.
    # This now includes base_lap_time_seconds to match the updated environment.
    env_params = {
        "total_laps": 58,
        "base_lap_time_seconds": 90.0,  # <<< ADDED FOR CLARITY & CONSISTENCY
        "pit_time": 28,
        "tire_wear_rate_config": 1.1,
        "traffic_penalty_config": 3.0
    }

    print("Creating training environment...")
    # Wrapper function to create an environment instance
    def make_env():
        env_instance = PitStopEnv(**env_params)
        env_instance = Monitor(env_instance) # Wrap with Monitor for SB3 logging
        return env_instance

    # PPO and other SB3 algorithms work best with VecEnvs (vectorized environments)
    env = DummyVecEnv([make_env])

    # Optional: Check if the environment conforms to the Gym API.
    # print("Checking environment (this might take a moment)...")
    # check_env(env.envs[0], warn=True) # Check the underlying non-vectorized environment

    # --- Model Definition ---
    # Define PPO hyperparameters
    tensorboard_log_path = "./ppo_pitstop_tensorboard/"
    os.makedirs(tensorboard_log_path, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        n_epochs=10,
        tensorboard_log=tensorboard_log_path
    )
    print(f"PPO Model Created. Observation Space: {env.observation_space}, Action Space: {env.action_space}")

    # --- Callbacks for enhanced training ---
    checkpoint_save_dir = './ppo_pitstop_checkpoints/'
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, model.n_steps * 5 // env.num_envs), # Save roughly every 5 rollouts
        save_path=checkpoint_save_dir,
        name_prefix='ppo_pitstop_model_ckpt'
    )

    eval_log_dir = './ppo_pitstop_eval_logs/'
    best_model_save_dir = './ppo_pitstop_best_model/'
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(best_model_save_dir, exist_ok=True)
    
    eval_env = DummyVecEnv([make_env]) # Create a separate instance for evaluation

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_dir,
        log_path=eval_log_dir,
        eval_freq=max(1, model.n_steps * 10 // env.num_envs), # Evaluate less frequently
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # --- Training ---
    total_timesteps_to_train = 300_000 
    print(f"Starting PPO training for {total_timesteps_to_train} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps_to_train,
            callback=[checkpoint_callback, eval_callback], # Use a list for multiple callbacks
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # --- Saving the Final Model ---
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_final_path = os.path.join(models_dir, "ppo_pit_stop.zip") 
        model.save(model_final_path)
        print(f"Training complete or interrupted. Final model saved to: {model_final_path}")

        best_model_zip_path = os.path.join(best_model_save_dir, "best_model.zip")
        if os.path.exists(best_model_zip_path):
            print(f"Best model during evaluation saved at: {best_model_zip_path}")
            # You might want to copy this best_model.zip to models/ppo_pit_stop.zip if it's better
            # import shutil
            # shutil.copy(best_model_zip_path, model_final_path)
            # print(f"Copied best model to {model_final_path}")
        else:
            print("No best model was saved by EvalCallback (or path is incorrect). Check evaluation logs.")

    # --- Post-Training: Manual Simulation & Reward Saving ---
    print("\nRunning one manual episode with the trained model to collect reward...")
    single_run_env = PitStopEnv(**env_params)
    obs, _ = single_run_env.reset()
    
    rewards_manual_run = []
    total_reward_manual = 0
    
    for _ in range(single_run_env.total_laps): 
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, terminated, truncated, _ = single_run_env.step(action)
        done = terminated or truncated
        total_reward_manual += reward
        if done:
            break
    rewards_manual_run.append(total_reward_manual)

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "ppo_rewards.npy"), np.array(rewards_manual_run))
    print(f"✅ PPO reward from one manual run saved to: {os.path.join(data_dir, 'ppo_rewards.npy')}")
    
    print("Closing environments.")
    env.close()
    eval_env.close()
    single_run_env.close()

if __name__ == '__main__':
    main()