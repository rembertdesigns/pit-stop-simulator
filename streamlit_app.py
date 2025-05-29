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

    # 🔍 Display lap-by-lap details
    st.subheader("📊 Lap-by-Lap Decisions")
    for lap, action, tire, traffic, reward in lap_data:
        st.write(f"Lap {lap}: {'PIT' if action else 'STAY OUT'} | Tire Wear: {tire:.2f} | Traffic: {traffic:.2f} | Reward: {reward:.2f}")

    pit_laps = [lap for lap, a, *_ in lap_data if a == 1]
    st.markdown(f"🚗 **Pit Stops:** {pit_laps if pit_laps else 'None'}")

    # 📈 Plot tire wear and reward with PIT markers
    laps = [lap for lap, *_ in lap_data]
    tire_wear = [tw for _, _, tw, _, _ in lap_data]
    rewards = [r for _, _, _, _, r in lap_data]

    st.subheader("📉 Tire Wear and Reward Over Time")
    fig, ax1 = plt.subplots(figsize=(10, 4))

    color1 = "tab:red"
    ax1.set_xlabel("Lap")
    ax1.set_ylabel("Tire Wear", color=color1)
    ax1.plot(laps, tire_wear, color=color1, label="Tire Wear")
    ax1.tick_params(axis='y', labelcolor=color1)

    # 🟠 Add PIT stop markers to tire wear plot
    pit_x = [lap for lap, a, *_ in lap_data if a == 1]
    pit_y = [tire_wear[i] for i, (lap, a, *_)
             in enumerate(lap_data) if a == 1]
    ax1.scatter(pit_x, pit_y, color="black", marker="x", s=80, label="PIT Stop")

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Reward", color=color2)
    ax2.plot(laps, rewards, color=color2, label="Reward")
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    st.pyplot(fig)