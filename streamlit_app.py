import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")

st.title("🏁 Pit Stop Strategy Simulator")

# Sidebar Settings
st.sidebar.header("🔧 Simulation Settings")
strategy = st.sidebar.selectbox("Choose a strategy", ["Q-learning", "PPO", "Head-to-Head"])
total_laps = st.sidebar.slider("Total Laps", 20, 70, 58)
pit_time = st.sidebar.slider("Pit Stop Time (s)", 20, 40, 30)
tire_wear_rate = st.sidebar.slider("Tire Wear Rate", 0.5, 2.5, 1.5)
traffic_penalty = st.sidebar.slider("Traffic Penalty Multiplier", 1.0, 10.0, 5.0)

# NEW: Team and Driver Context
st.sidebar.subheader("🏎️ Team & Driver Context")
teams = {
    "Red Bull": {"tire_wear_rate": 1.3, "traffic_penalty": 4.5},
    "Mercedes": {"tire_wear_rate": 1.5, "traffic_penalty": 5.0},
    "Ferrari": {"tire_wear_rate": 1.7, "traffic_penalty": 5.5},
    "McLaren": {"tire_wear_rate": 1.6, "traffic_penalty": 5.2},
    "Aston Martin": {"tire_wear_rate": 1.4, "traffic_penalty": 5.3},
}
selected_team = st.sidebar.selectbox("Choose a team", list(teams.keys()))

def create_env():
    settings = teams[selected_team]
    return PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=settings["tire_wear_rate"],
        traffic_penalty=settings["traffic_penalty"]
    )

def run_simulation(strategy_name):
    env = create_env()
    lap_data = []

    if strategy_name == "Q-learning":
        agent = QAgent(env)
        obs, _ = env.reset()
        state = agent.discretize(obs)
        for _ in range(total_laps):
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = agent.discretize(next_obs)
            lap_data.append((env.current_lap, action, next_obs[1], next_obs[2], reward))
            if terminated or truncated:
                break

    elif strategy_name == "PPO":
        model = PPO.load("models/ppo_pit_stop")
        obs, _ = env.reset()
        for _ in range(total_laps):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            lap_data.append((env.current_lap, action, obs[1], obs[2], reward))
            if terminated or truncated:
                break

    return lap_data

def animate_lap_chart(lap_data, strategy_name):
    fig, ax = plt.subplots()
    tire_wear_line, = ax.plot([], [], label='Tire Wear')
    traffic_line, = ax.plot([], [], label='Traffic')
    pit_markers = []

    ax.set_xlim(0, total_laps)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Value")
    ax.set_title(f"Lap-by-Lap Simulation: {strategy_name}")
    ax.legend()

    tire_wear_vals = []
    traffic_vals = []
    x_vals = []

    chart_placeholder = st.empty()

    for lap, action, tire, traffic, reward in lap_data:
        x_vals.append(lap)
        tire_wear_vals.append(tire)
        traffic_vals.append(traffic * 100)

        tire_wear_line.set_data(x_vals, tire_wear_vals)
        traffic_line.set_data(x_vals, traffic_vals)

        if action == 1:
            ax.axvline(x=lap, color='red', linestyle='--', linewidth=0.8)

        chart_placeholder.pyplot(fig)
        time.sleep(0.1)

    st.markdown(f"🚗 **Pit Stops:** {[lap for lap, a, *_ in lap_data if a == 1]}")

# Run Simulations
if strategy != "Head-to-Head":
    lap_data = run_simulation(strategy)
    st.subheader(f"📊 {strategy} Strategy Replay")
    animate_lap_chart(lap_data, strategy)
else:
    st.subheader("⚔️ Head-to-Head Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Q-learning")
        q_data = run_simulation("Q-learning")
        animate_lap_chart(q_data, "Q-learning")

    with col2:
        st.markdown("### PPO")
        ppo_data = run_simulation("PPO")
        animate_lap_chart(ppo_data, "PPO")
