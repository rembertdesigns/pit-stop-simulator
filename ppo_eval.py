import os
import numpy as np
import gymnasium as gym # Using Gymnasium standard
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor # Good for consistent episode stats

# Ensure this path correctly points to your updated PitStopEnv
from env.gym_race_env import PitStopEnv

def main():
    print("🏁 Starting PPO Model Evaluation...")

    # --- Environment Setup ---
    # Use similar parameters as training or specific evaluation parameters if needed
    env_params = {
        "total_laps": 58,
        "pit_time": 28,
        "tire_wear_rate_config": 1.1,
        "traffic_penalty_config": 3.0
    }
    env = PitStopEnv(**env_params)
    env = Monitor(env) # Wrap with Monitor for consistent episode data (optional for simple eval)

    # --- Load the Retrained Model ---
    model_path = "models/ppo_pit_stop.zip" # Explicitly use .zip
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Please train the PPO model first.")
        return

    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading PPO model: {e}")
        return

    # --- Evaluation Parameters ---
    total_episodes_to_evaluate = 5 # Evaluate over a few episodes for better average
    all_episode_rewards = []
    all_episode_pit_decisions = [] # To store pit decisions for all episodes

    print(f"\nRunning evaluation for {total_episodes_to_evaluate} episode(s)...")

    for episode in range(total_episodes_to_evaluate):
        print(f"\n--- Episode {episode + 1}/{total_episodes_to_evaluate} ---")
        obs, info = env.reset() # info might be available from Monitor or custom env
        done = False
        current_episode_total_reward = 0
        current_episode_pit_laps = []
        lap_counter = 0 # Manual lap counter for clarity in print

        while not done:
            lap_counter += 1
            # Use deterministic=True for consistent evaluation
            action, _ = model.predict(obs, deterministic=True)
            
            # Store pit decision *before* the step, using env.current_lap if available
            # env.current_lap is 0-indexed before the first step, then updated.
            # obs[0] is the lap number *after* the previous step (or initial lap)
            # For clarity, let's use the lap number about to be simulated
            lap_being_simulated = env.current_lap # Or obs[0] + 1 if using obs from previous step for lap num

            if action == 1:
                # Log the lap number on which the decision to pit is made
                current_episode_pit_laps.append(lap_being_simulated)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_episode_total_reward += reward

            # env.render() # If you want to see the human render method from PitStopEnv
            print(f"  Lap {lap_being_simulated:>2} (Obs Lap: {int(obs[0]):>2}) | Action: {'PIT    ' if action == 1 else 'STAYOUT'} | "
                  f"TireWear: {env.get_attr('tire_wear')[0]:6.2f}% | Traffic: {env.get_attr('traffic')[0]:.2f} | " # Use get_attr for VecEnv/Monitor
                  f"Fuel: {env.get_attr('fuel_weight')[0]:6.2f}kg | Rain: {env.get_attr('rain_active')[0]} | SC: {env.get_attr('safety_car_active')[0]} | "
                  f"Reward: {reward:7.2f} | Cum. Reward: {current_episode_total_reward:8.2f}")
            
            if done:
                print(f"--- Episode {episode + 1} Finished ---")
                print(f"  Total Reward for Episode: {current_episode_total_reward:.2f}")
                print(f"  Pit stops occurred on (decision) laps: {current_episode_pit_laps if current_episode_pit_laps else 'None'}")
                break
        
        all_episode_rewards.append(current_episode_total_reward)
        all_episode_pit_decisions.append(current_episode_pit_laps)

    print("\n✅ Evaluation Complete.")
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    print(f"\n--- Summary over {total_episodes_to_evaluate} episodes ---")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("Rewards per episode:", [round(r, 2) for r in all_episode_rewards])
    print("Pit decisions per episode:", all_episode_pit_decisions)


    # --- Save Evaluation Data ---
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    rewards_save_path = os.path.join(data_dir, "ppo_eval_rewards.npy")
    np.save(rewards_save_path, np.array(all_episode_rewards))
    print(f"\n✅ PPO evaluation rewards saved to: {rewards_save_path}")

    # Optionally save pit decisions
    # Note: pit_decisions is a list of lists, which np.save handles but might need padding if saving as a strict 2D array.
    # Saving as a .pkl or .json might be better for lists of varying length lists.
    # For simplicity with numpy:
    try:
        # This will save it as an object array if sub-lists have different lengths
        pit_decisions_save_path = os.path.join(data_dir, "ppo_eval_pit_decisions.npy")
        np.save(pit_decisions_save_path, np.array(all_episode_pit_decisions, dtype=object))
        print(f"✅ PPO evaluation pit decisions saved to: {pit_decisions_save_path}")
    except Exception as e:
        print(f"Could not save pit decisions as .npy: {e}. Consider saving as .pkl or .json instead.")

    env.close()

if __name__ == '__main__':
    main()