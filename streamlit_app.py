import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")
st.title("🏁 Pit Stop Strategy Simulator")

# --- Shared Controls ---
with st.sidebar:
    st.header("🛠 Simulation Settings")
    total_laps = st.slider("Total Laps", 20, 70, 58)
    pit_time = st.slider("Pit Stop Time (seconds)", 10, 40, 30)
    tire_degradation = st.slider("Tire Wear Rate (per lap)", 0.5, 3.0, 1.5, step=0.1)
    traffic_penalty = st.slider("Traffic Penalty Multiplier", 0.0, 10.0, 5.0, step=0.5)

# --- Strategy Runner Functions ---
def run_q_learning():
    env = PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=tire_degradation,
        traffic_factor=traffic_penalty,
    )
    agent = QAgent(env)
    obs, _ = env.reset()
    state = agent.discretize(obs)
    data = []
    for _ in range(total_laps):
        action = agent.choose_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state = agent.discretize(next_obs)
        data.append((env.current_lap, action, next_obs[1], next_obs[2], reward))
        if terminated or truncated:
            break
    return data

def run_ppo():
    env = PitStopEnv(
        total_laps=total_laps,
        pit_time=pit_time,
        tire_wear_rate=tire_degradation,
        traffic_factor=traffic_penalty,
    )
    model = PPO.load("models/ppo_pit_stop")
    obs, _ = env.reset()
    data = []
    for _ in range(total_laps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        data.append((env.current_lap, action, obs[1], obs[2], reward))
        if terminated or truncated:
            break
    return data

# --- Plot Function ---
def plot_strategy_chart(data, label):
    laps = [lap for lap, *_ in data]
    rewards = [r for *_, r in data]
    pit_laps = [lap for lap, action, *_ in data if action == 1]

    fig, ax = plt.subplots()
    ax.plot(laps, rewards, label="Reward", color="blue")
    for lap in pit_laps:
        ax.axvline(x=lap, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Reward")
    ax.set_title(f"{label} Strategy")
    ax.legend()
    st.pyplot(fig)

# --- Run & Display ---
if st.button("Simulate & Compare"):
    q_data = run_q_learning()
    ppo_data = run_ppo()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Q-learning")
        plot_strategy_chart(q_data, "Q-learning")
        st.markdown("**Lap Decisions**")
        for lap, act, tire, traffic, reward in q_data:
            st.write(f"Lap {lap}: {'PIT' if act else 'STAY OUT'} | Tire: {tire:.2f} | Traffic: {traffic:.2f} | Reward: {reward:.2f}")
        q_pits = [lap for lap, a, *_ in q_data if a == 1]
        st.markdown(f"🚗 Pit Stops: {q_pits if q_pits else 'None'}")

    with col2:
        st.subheader("🧠 PPO")
        plot_strategy_chart(ppo_data, "PPO")
        st.markdown("**Lap Decisions**")
        for lap, act, tire, traffic, reward in ppo_data:
            st.write(f"Lap {lap}: {'PIT' if act else 'STAY OUT'} | Tire: {tire:.2f} | Traffic: {traffic:.2f} | Reward: {reward:.2f}")
        ppo_pits = [lap for lap, a, *_ in ppo_data if a == 1]
        st.markdown(f"🚗 Pit Stops: {ppo_pits if ppo_pits else 'None'}")