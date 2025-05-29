import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent
from stable_baselines3 import PPO

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")

st.title("🏁 Pit Stop Strategy Simulator")

strategy = st.selectbox("Choose a strategy", ["Q-learning", "PPO"])
total_laps = st.slider("Total Laps", 20, 70, 58)

env = PitStopEnv(total_laps=total_laps)

if st.button("Simulate"):
    lap_data = []

    if strategy == "Q-learning":
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

    else:  # PPO
        model = PPO.load("models/ppo_pit_stop")
        obs, _ = env.reset()
        for _ in range(total_laps):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            lap_data.append((env.current_lap, action, obs[1], obs[2], reward))
            if terminated or truncated:
                break

    st.subheader("📊 Lap-by-Lap Decisions")
    for lap, action, tire, traffic, reward in lap_data:
        st.write(f"Lap {lap}: {'PIT' if action else 'STAY OUT'} | Tire Wear: {tire:.2f} | Traffic: {traffic:.2f} | Reward: {reward:.2f}")

    pit_laps = [lap for lap, a, *_ in lap_data if a == 1]
    st.markdown(f"🚗 **Pit Stops:** {pit_laps if pit_laps else 'None'}")
