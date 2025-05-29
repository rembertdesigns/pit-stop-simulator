import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")
st.title("🏁 Pit Stop Strategy Simulator")

# --- Controls ---
total_laps = st.slider("Total Laps", 20, 70, 58)
compare_mode = st.checkbox("🔍 Compare Q-learning vs PPO", value=True)

env = PitStopEnv(total_laps=total_laps)

def run_q_learning(env):
    agent = QAgent(env)
    obs, _ = env.reset()
    state = agent.discretize(obs)
    lap_data = []
    for _ in range(total_laps):
        action = agent.choose_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state = agent.discretize(next_obs)
        lap_data.append((env.current_lap, action, next_obs[1], next_obs[2], reward))
        if terminated or truncated:
            break
    return lap_data

def run_ppo(env):
    model = PPO.load("models/ppo_pit_stop")
    obs, _ = env.reset()
    lap_data = []
    for _ in range(total_laps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        lap_data.append((env.current_lap, action, obs[1], obs[2], reward))
        if terminated or truncated:
            break
    return lap_data

def plot_lap_chart(lap_data, label):
    laps = [lap for lap, *_ in lap_data]
    tire_wear = [tw for _, _, tw, _, _ in lap_data]
    pit_laps = [lap for lap, action, *_ in lap_data if action == 1]

    fig, ax = plt.subplots()
    ax.plot(laps, tire_wear, label="Tire Wear", linewidth=2)
    for lap in pit_laps:
        ax.axvline(x=lap, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Tire Wear")
    ax.set_title(f"{label} Strategy")
    ax.legend()
    st.pyplot(fig)

if st.button("Simulate"):

    if compare_mode:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 Q-learning")
            q_env = PitStopEnv(total_laps=total_laps)
            q_data = run_q_learning(q_env)
            plot_lap_chart(q_data, "Q-learning")
            q_pits = [lap for lap, action, *_ in q_data if action == 1]
            st.markdown(f"🚗 Pit Laps: {q_pits if q_pits else 'None'}")

        with col2:
            st.subheader("🧠 PPO")
            ppo_env = PitStopEnv(total_laps=total_laps)
            ppo_data = run_ppo(ppo_env)
            plot_lap_chart(ppo_data, "PPO")
            ppo_pits = [lap for lap, action, *_ in ppo_data if action == 1]
            st.markdown(f"🚗 Pit Laps: {ppo_pits if ppo_pits else 'None'}")

    else:
        st.warning("Enable comparison mode above to view both strategies side-by-side.")