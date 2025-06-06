# 🏎️ Pit Stop Optimization Simulator

A reinforcement learning simulation to optimize Formula 1 pit stop strategies based on tire wear, traffic conditions, and lap count. Built using a custom `SimPy` and `Gym` environment, and trained with a tabular Q-learning agent.

---

## 🎯 Objective

Teach an agent to decide **when to pit** during a race to minimize lap times and maximize performance under dynamic conditions.

---

## 🧱 Project Structure
```bash
pit-stop-simulator/
│
├── main.py                      # Q-learning training loop + pit heatmap
├── ppo_train.py                 # Train PPO agent using Stable-Baselines3
├── ppo_eval.py                  # Evaluate PPO agent and track pit stops
├── compare_strategies.py        # Visual side-by-side comparison of pit strategies
├── streamlit_app.py             # Interactive Streamlit UI with race events and agent simulations
│
├── env/
│   └── gym_race_env.py          # Custom OpenAI Gymnasium race simulation environment
│
├── rl/
│   └── q_learning_agent.py      # Q-learning agent logic
│
├── models/
│   └── ppo_pit_stop.zip         # Saved PPO model
│
├── saved_agents/                # Saved Q-learning agents (per team/profile)
│
├── data/
│   ├── q_learning_pit_decisions.npy   # Pit stop logs from Q-learning
│   └── ppo_pit_decisions.npy          # Pit stop logs from PPO
│
├── requirements.txt             # Python package dependencies
└── README.md                    # Project overview and setup instructions
```
---

## 🚀 How It Works

- **Environment** simulates 58 laps with dynamic traffic and tire wear
- **Actions**: `0 = stay out`, `1 = pit`
- **Observations**: `[lap number, tire wear, traffic level]`
- **Rewards**: Negative lap times (shorter = better)

---

## 🤖 Agent Logic (Q-learning)

- Discretizes continuous state space (lap, tire, traffic)
- Balances exploration vs. exploitation using ε-greedy strategy
- Updates Q-values over episodes to learn optimal pit timing

---

## 📊 Visualization

After training, a reward curve is plotted showing the agent's learning progress over time.

---

## 🧪 Streamlit Simulation Dashboard

The interactive UI lets you:

- 🔁 Replay Animated Laps with ghost trails
- 📊 Visualizations: Tire wear, traffic, fuel, grip, temperature
- 🌧️ Probabilistic Rain Forecasts by lap range & chance %
- 🛞 Tire Compounds: Soft, Medium, Hard, Intermediate, Wet
- 🚨 Dynamic Events: Safety Cars, VSCs, Crashes
- 🧠 Driver Profiles: Aggressive, Balanced, Conservative
- 🛠️ Custom Strategy Mode: Pick your own pit stops
- 📈 ML Lap Time Predictor: Trained with XGBoost
- 🧾 PDF Race Summary Reports for download
- 🧠 Strategic Timeline with emoji markers:
- 🅿️ Pit Stop
- 🌧️ Rain
- ⚠️ Safety Car

---

## 📦 Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training script
python3 main.py

# Launch the Streamlit simulator
streamlit run streamlit_app.py
```

---

## 🛠️ Roadmap

- Visualize pit stop decisions (lap heatmaps)
- Add multiple tire compounds
- Introduce safety car periods or weather
- Switch to PPO (for continuous observations)
- Add real-time event adaptation
- Support persistent driver learning and history tracking
- Enable full race replay controls

 ---
 
## 📄 License

MIT — use, modify, and share freely.

---

🙌 Credits

Built with:

- SimPy
- OpenAI Gym
- NumPy
- Matplotlib
