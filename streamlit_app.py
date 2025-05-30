import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

st.set_page_config(page_title="🏁 Pit Stop Strategy Simulator", layout="wide")
st.title("🏁 Head-to-Head Strategy Comparison: PPO vs Q-learning")

# Sidebar controls
st.sidebar.header("Simulation Settings")
total_laps = st.sidebar.slider("Total Laps", 20, 70, 58)
pit_time = st.sidebar.slider("Pit Stop Time Penalty", 10, 60, 30)
tire_wear_rate = st.sidebar.slider("Tire Wear Rate", 0.5, 3.0, 1.5)
traffic_penalty = st.sidebar.slider("Traffic Penalty Factor", 0.0, 10.0, 5.0)
speed = st.sidebar.slider("Animation Speed (seconds per lap)", 0.05, 1.0, 0.3)

# Environment Factory
def create_env():
    env = PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=tire_wear_rate,
        traffic_penalty=traffic_penalty
    )
    return env

# Run a single lap-by-lap simulation for a strategy
def simulate_strategy(strategy, env):
    lap_data = []
    if strategy == "Q-learning":
        agent = QAgent(env)
        obs, _ = env.reset()
        state = agent.discretize(obs)
        for _ in range(env.total_laps):
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = agent.discretize(next_obs)
            lap_data.append((env.current_lap, action, next_obs[1], next_obs[2], reward))
            if terminated or truncated:
                break
    else:  # PPO
        model = PPO.load("models/ppo_pit_stop")
        obs, _ = env.reset()
        for _ in range(env.total_laps):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            lap_data.append((env.current_lap, action, obs[1], obs[2], reward))
            if terminated or truncated:
                break
    return lap_data

# Animate side-by-side strategies
def animate_comparison():
    q_env = create_env()
    ppo_env = create_env()
    q_data = simulate_strategy("Q-learning", q_env)
    ppo_data = simulate_strategy("PPO", ppo_env)

    q_rewards, ppo_rewards = [], []
    q_pits, ppo_pits = [], []

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Lap-by-Lap Total Reward Comparison")
    ax.set_xlim(0, total_laps)
    ax.set_ylim(-1600, 0)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Cumulative Reward")

    q_line, = ax.plot([], [], label="Q-learning", color="blue")
    ppo_line, = ax.plot([], [], label="PPO", color="orange")
    ax.legend()

    plot_placeholder = st.empty()
    log_placeholder = st.empty()

    for lap in range(total_laps):
        # Add rewards
        if lap < len(q_data):
            q_rewards.append(sum(r for _, _, _, _, r in q_data[:lap+1]))
            if q_data[lap][1] == 1:
                q_pits.append(lap)
        else:
            q_rewards.append(q_rewards[-1] if q_rewards else 0)

        if lap < len(ppo_data):
            ppo_rewards.append(sum(r for _, _, _, _, r in ppo_data[:lap+1]))
            if ppo_data[lap][1] == 1:
                ppo_pits.append(lap)
        else:
            ppo_rewards.append(ppo_rewards[-1] if ppo_rewards else 0)

        q_line.set_data(range(lap+1), q_rewards)
        ppo_line.set_data(range(lap+1), ppo_rewards)

        ax.scatter(q_pits, [q_rewards[i] for i in q_pits], marker='X', color='blue', label='_nolegend_')
        ax.scatter(ppo_pits, [ppo_rewards[i] for i in ppo_pits], marker='X', color='orange', label='_nolegend_')

        plot_placeholder.pyplot(fig)
        log_placeholder.markdown(
            f"**Lap {lap + 1}** | Q-learning Cumulative Reward: `{q_rewards[-1]:.2f}` | PPO Cumulative Reward: `{ppo_rewards[-1]:.2f}`"
        )
        time.sleep(speed)

    final_q = q_rewards[-1]
    final_ppo = ppo_rewards[-1]
    st.success(f"🏁 Final Reward — Q-learning: `{final_q:.2f}` | PPO: `{final_ppo:.2f}`")
    st.markdown(f"🚗 **Q-learning Pit Stops:** {q_pits if q_pits else 'None'}")
    st.markdown(f"🏎️ **PPO Pit Stops:** {ppo_pits if ppo_pits else 'None'}")

# Run on click
if st.button("Run Head-to-Head Simulation"):
    animate_comparison()
