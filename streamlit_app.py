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
strategy = st.sidebar.selectbox("Choose a strategy", ["Q-learning", "PPO", "Head-to-Head", "Custom Strategy"])
total_laps = st.sidebar.slider("Total Laps", 20, 70, 58)

# Track Selection with auto-parameters
track_options = {
    "Spa": {"pit_time": 32, "tire_wear_rate": 1.2, "traffic_penalty": 4.0},
    "Monaco": {"pit_time": 25, "tire_wear_rate": 1.4, "traffic_penalty": 7.5},
    "Bahrain": {"pit_time": 30, "tire_wear_rate": 2.0, "traffic_penalty": 5.0},
    "Custom": {}  # for manual override
}
selected_track = st.sidebar.selectbox("Choose a Track", list(track_options.keys()))

if selected_track == "Custom":
    pit_time = st.sidebar.slider("Pit Stop Time (s)", 20, 40, 30)
    tire_wear_rate = st.sidebar.slider("Tire Wear Rate", 0.5, 2.5, 1.5)
    traffic_penalty = st.sidebar.slider("Traffic Penalty Multiplier", 1.0, 10.0, 5.0)
else:
    track = track_options[selected_track]
    pit_time = track["pit_time"]
    tire_wear_rate = track["tire_wear_rate"]
    traffic_penalty = track["traffic_penalty"]

# Team and Driver Context
st.sidebar.subheader("🏎️ Team & Driver Context")
teams = {
    "Red Bull": {"color": "#1E41FF", "logo": "🟥 Verstappen"},
    "Mercedes": {"color": "#00D2BE", "logo": "⬛ Hamilton"},
    "Ferrari": {"color": "#DC0000", "logo": "🟥 Leclerc"},
    "McLaren": {"color": "#FF8700", "logo": "🟧 Norris"},
    "Aston Martin": {"color": "#006F62", "logo": "🟩 Alonso"},
}
selected_team = st.sidebar.selectbox("Choose a team", list(teams.keys()))

# Driver Profiles
driver_profiles = {
    "Aggressive": {"pit_threshold": 80, "bonus": 2},
    "Balanced": {"pit_threshold": 65, "bonus": 1},
    "Conservative": {"pit_threshold": 50, "bonus": 0}
}
selected_profile = st.sidebar.selectbox("Driver Profile", list(driver_profiles.keys()))

# Custom Strategy Builder
custom_pit_laps = []
if strategy == "Custom Strategy":
    st.sidebar.subheader("🛠️ Custom Strategy")
    selected_laps = st.sidebar.multiselect("Select Pit Laps", list(range(1, total_laps + 1)))
    custom_pit_laps = set(selected_laps)

start_button = st.button("▶️ Start Simulation")

# Environment Creation
def create_env():
    return PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=tire_wear_rate,
        traffic_penalty=traffic_penalty
    )

# Simulation Logic
def run_simulation(strategy_name):
    env = create_env()
    lap_data = []
    profile = driver_profiles[selected_profile]
    obs, _ = env.reset()

    if strategy_name == "Q-learning":
        agent = QAgent(env)
        state = agent.discretize(obs)
        for _ in range(total_laps):
            action = agent.choose_action(state)
            if obs[1] > profile["pit_threshold"]:
                action = 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = agent.discretize(next_obs)
            lap_data.append((env.current_lap, action, next_obs[1], next_obs[2], reward + profile["bonus"]))
            obs = next_obs
            if terminated or truncated:
                break

    elif strategy_name == "PPO":
        model = PPO.load("models/ppo_pit_stop")
        for _ in range(total_laps):
            action, _ = model.predict(obs)
            if obs[1] > profile["pit_threshold"]:
                action = 1
            obs, reward, terminated, truncated, _ = env.step(action)
            lap_data.append((env.current_lap, action, obs[1], obs[2], reward + profile["bonus"]))
            if terminated or truncated:
                break

    elif strategy_name == "Custom Strategy":
        for _ in range(total_laps):
            action = 1 if env.current_lap in custom_pit_laps else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            lap_data.append((env.current_lap, action, obs[1], obs[2], reward))
            if terminated or truncated:
                break

    return lap_data

# Animated Chart
def animate_lap_chart(lap_data, strategy_name, color):
    fig, ax = plt.subplots()
    tire_wear_line, = ax.plot([], [], label='Tire Wear (%)', color=color)
    traffic_line, = ax.plot([], [], label='Traffic Intensity (%)', linestyle='--', color='gray')

    ax.set_xlim(0, total_laps)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Simulation Metric (%)")
    ax.set_title(f"{selected_team} | {selected_profile} Driver: {strategy_name}")
    ax.legend()

    tire_vals, traffic_vals, x_vals = [], [], []
    chart_placeholder = st.empty()

    for lap, action, tire, traffic, reward in lap_data:
        x_vals.append(lap)
        tire_vals.append(tire)
        traffic_vals.append(traffic * 100)

        tire_wear_line.set_data(x_vals, tire_vals)
        traffic_line.set_data(x_vals, traffic_vals)

        if action == 1:
            ax.axvline(x=lap, color='red', linestyle='--', linewidth=0.8)

        chart_placeholder.pyplot(fig)
        time.sleep(0.1)

    st.markdown(f"🚗 **Pit Stops:** {[lap for lap, a, *_ in lap_data if a == 1]}")

# Run Simulation
if start_button:
    if strategy != "Head-to-Head":
        lap_data = run_simulation(strategy)
        st.subheader(f"📊 {strategy} Strategy Replay")
        animate_lap_chart(lap_data, strategy, teams[selected_team]["color"])
    else:
        st.subheader("⚔️ Head-to-Head Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Q-learning")
            q_data = run_simulation("Q-learning")
            animate_lap_chart(q_data, "Q-learning", teams[selected_team]["color"])

        with col2:
            st.markdown("### PPO")
            ppo_data = run_simulation("PPO")
            animate_lap_chart(ppo_data, "PPO", teams[selected_team]["color"])