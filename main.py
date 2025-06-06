# main.py (Enhanced for Batch Training Q-Learning Agents)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # For a nice progress bar

# Assuming these custom modules are in the correct path and are updated
from env.gym_race_env import PitStopEnv 
from rl.q_learning_agent import QAgent 

# --- 1. CONFIGURATION ---

# Define all team/profile combinations you want to train agents for.
# These names MUST EXACTLY MATCH the keys in your streamlit_app.py dictionaries.
TEAMS_TO_TRAIN = ["Ferrari", "Red Bull", "Mercedes", "McLaren", "Aston Martin"]
PROFILES_TO_TRAIN = ["Aggressive", "Balanced", "Conservative"]

# Define general training parameters
TRAINING_EPISODES = 2000 # Increased for more thorough learning
FIGURES_DIR = "training_figures" # Directory to save output plots

# Define parameters for the training environment.
# This ensures agents are trained on a consistent, representative track.
TRAINING_ENV_PARAMS = {
    "total_laps": 58,
    "base_lap_time_seconds": 90.0, # e.g., A Monza-like pace
    "pit_time": 28,
    "tire_wear_rate_config": 1.1,
    "traffic_penalty_config": 3.0
}


# --- 2. TRAINING & SAVING FUNCTION ---

def train_and_save_agent(team, profile, episodes, env_params):
    """
    Initializes, trains, saves, and plots results for a single Q-learning agent.
    """
    print(f"\n--- Initializing training for: {team} - {profile} ---")
    
    # Initialize Environment and Agent for this specific run
    env = PitStopEnv(**env_params)
    agent = QAgent(env) # QAgent is initialized with the env (and its 6D action space)
    
    rewards_per_episode = []
    pit_decisions_per_episode = []

    # Use tqdm for a live progress bar in the terminal
    for episode in tqdm(range(episodes), desc=f"Training {team} {profile}"):
        obs, _ = env.reset()
        state = agent.discretize(obs)
        total_reward = 0
        done = False
        current_episode_pit_laps = []

        while not done:
            action = agent.choose_action(state) # Agent chooses action 0-5
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize(next_obs)
            
            if action > 0: # Any action > 0 is a pit stop
                current_episode_pit_laps.append(int(env.current_lap))

            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        pit_decisions_per_episode.append(current_episode_pit_laps)
    
    print(f"--- Training for {team} - {profile} complete ---")
    
    # --- Save the Agent ---
    saved_agents_dir = "saved_agents"
    os.makedirs(saved_agents_dir, exist_ok=True)
    save_path = os.path.join(saved_agents_dir, f"{team}_{profile}_q.pkl")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(agent, f)
        print(f"✅ Agent successfully saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving agent: {e}")

    # --- Generate and Save Plots ---
    plot_rewards(rewards_per_episode, team, profile)
    plot_pit_heatmap(pit_decisions_per_episode, env_params["total_laps"], team, profile)


# --- 3. PLOTTING FUNCTIONS ---

def plot_rewards(rewards, team, profile):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Total Reward per Episode')
    
    rolling_avg_window = max(1, len(rewards) // 10) # e.g., 10% of episodes
    if len(rewards) > rolling_avg_window:
        rolling_avg = np.convolve(rewards, np.ones(rolling_avg_window)/rolling_avg_window, mode='valid')
        plt.plot(np.arange(rolling_avg_window - 1, len(rewards)), rolling_avg, 
                 label=f'{rolling_avg_window}-Episode Rolling Average', color='orange', linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Higher is Better)")
    plt.title(f"Q-learning Training Progress: {team} - {profile}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"rewards_{team}_{profile}.png"))
    plt.close() # Close figure to free up memory

def plot_pit_heatmap(pit_decisions, total_laps, team, profile, bins=10):
    if not any(pit_decisions):
        print(f"No pit stops were made for {team}-{profile}, cannot generate heatmap.")
        return
        
    num_episodes = len(pit_decisions)
    if num_episodes < bins: bins = max(1, num_episodes)
    heatmap = np.zeros((bins, total_laps))
    episodes_per_bin = max(1, num_episodes // bins)

    for b in range(bins):
        start_index = b * episodes_per_bin
        end_index = (b + 1) * episodes_per_bin
        for episode_pits in pit_decisions[start_index:end_index]:
            for lap in episode_pits:
                if lap < total_laps: heatmap[b, lap] += 1

    plt.figure(figsize=(14, 7))
    sns.heatmap(heatmap, cmap="YlGnBu", xticklabels=5, 
                yticklabels=[f"Ep {i*episodes_per_bin}-{(i+1)*episodes_per_bin}" for i in range(bins)])
    plt.xlabel("Lap Number")
    plt.ylabel("Training Progress (Episodes)")
    plt.title(f"Pit Stop Frequency Heatmap: {team} - {profile}")
    plt.tight_layout()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"heatmap_{team}_{profile}.png"))
    plt.close()


# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Loop through all configured teams and profiles to train an agent for each
    for team_name in TEAMS_TO_TRAIN:
        for profile_name in PROFILES_TO_TRAIN:
            train_and_save_agent(
                team=team_name,
                profile=profile_name,
                episodes=TRAINING_EPISODES,
                env_params=TRAINING_ENV_PARAMS
            )

    print("\n===== ALL Q-LEARNING AGENT TRAINING COMPLETE =====")