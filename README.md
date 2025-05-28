# 🏎️ Pit Stop Optimization Simulator

A reinforcement learning simulation to optimize Formula 1 pit stop strategies based on tire wear, traffic conditions, and lap count. Built using a custom `SimPy` and `Gym` environment, and trained with a tabular Q-learning agent.

---

## 🎯 Objective

Teach an agent to decide **when to pit** during a race to minimize lap times and maximize performance under dynamic conditions.

---

## 🧱 Project Structure
```bash
PIT-STOP-SIMULATOR/
├── env/ # Custom Gym-compatible environment
│ └── gym_race_env.py
├── rl/ # Reinforcement learning agents
│ └── q_learning_agent.py
├── data/ # Placeholder for track/tire data
├── visualizations/ # Future visual outputs (heatmaps, charts)
├── main.py # Entry point: trains and visualizes agent
├── requirements.txt # Dependencies
└── README.md
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

## 📦 Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training script
python3 main.py
```

---

## 🛠️ Roadmap

 - Visualize pit stop decisions (lap heatmaps)
 - Add multiple tire compounds
 - Introduce safety car periods or weather
 - Switch to PPO (for continuous observations)

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
