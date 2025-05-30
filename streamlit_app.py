import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time

from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏁 Pit Stop Strategy Simulator (Animated Replay)")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Settings")
strategy = st.sidebar.selectbox("Choose Strategy", ["Q-learning", "PPO"])
total_laps = st.sidebar.slider("Total Laps", 10, 70, 40)
pit_time = st.sidebar.slider("Pit Stop Time (seconds)", 20, 40, 30)
tire_wear_rate = st.sidebar.slider("Tire Wear per Lap", 0.5, 3.0, 1.5)
traffic_penalty = st.sidebar.slider("Traffic Penalty Multiplier", 0.0, 10.0, 5.0)

start = st.sidebar.button("▶️ Run Simulation")

# --- Core Logic ---
def run_simulation():
    env = PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=tire_wear_rate,
        traffic_penalty=traffic_penalty,
    )
    rewards = []
    pit_laps = []
    
    if strategy == "Q-learning":
        agent = QAgent(env)
        obs, _ = env.reset()
        state = agent.discretize(obs)
        for _ in range(total_laps):
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = agent.discretize(next_obs)
            rewards.append(reward)
            if action == 1:
                pit_laps.append(env.current_lap)
            if terminated or truncated:
                break
    else:
        model = PPO.load("models/ppo_pit_stop")
        obs, _ = env.reset()
        for _ in range(total_laps):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if action == 1:
                pit_laps.append(env.current_lap)
            if terminated or truncated:
                break
    return rewards, pit_laps

# --- Visualization ---
if start:
    st.subheader(f"📉 {strategy} Strategy: Lap-by-Lap Rewards")
    chart = st.empty()
    status = st.empty()

    rewards, pit_laps = run_simulation()
    for lap in range(1, len(rewards) + 1):
        fig, ax = plt.subplots()
        ax.plot(range(1, lap + 1), rewards[:lap], color="blue", label="Reward")

        for pit in pit_laps:
            if pit <= lap:
                ax.axvline(pit, color="red", linestyle="--", linewidth=1)

        ax.set_xlim(1, total_laps)
        ax.set_ylim(min(rewards) - 5, max(rewards) + 5)
        ax.set_xlabel("Lap")
        ax.set_ylabel("Reward")
        ax.set_title("Live Lap Reward Over Time")
        ax.grid(True)
        ax.legend()

        chart.pyplot(fig)
        current_action = "🚗 PIT" if lap in pit_laps else "STAY OUT"
        status.markdown(f"**Lap {lap}:** {current_action}")
        time.sleep(0.2)

    st.success("✅ Simulation Complete!")
