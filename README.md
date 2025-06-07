# 🏎️ F1 Pit Stop Strategy Simulator

An advanced, interactive simulator built with Python, Streamlit, and Reinforcement Learning (`Q-learning` & `PPO`) to explore and optimize Formula 1 pit stop strategies.

This application simulates full F1 races, considering dynamic conditions including non-linear tire degradation, track characteristics, fuel load, traffic, probabilistic weather (rain), and on-track incidents like Safety Cars and Virtual Safety Cars (VSC).

---

## ✨ Key Features

### Dynamic Race Environment
A custom-built environment using `gymnasium` that simulates:

- Lap-by-lap race progression with track-specific base lap times.
- **Advanced Tire Model**: Non-linear degradation profiles for multiple compounds (Soft, Medium, Hard) and wet-weather tires (Intermediate, Wet), each with unique wear rates and performance drop-offs.
- **Fuel consumption** and its impact on lap times.
- Variable **track grip** and **temperature evolution**.
- **Randomized traffic** events affecting lap times.

### Intelligent Strategy Agents

- **Q-Learning Agent**: A custom-built agent that learns optimal pit strategies by discretizing the complex state space.
- **PPO Agent**: Utilizes Proximal Policy Optimization from `stable-baselines3` for more advanced strategy learning.
- **Expanded Action Space**: Agents can make granular decisions, choosing not just when to pit, but which specific tire compound to switch to.

### Comprehensive Streamlit Dashboard

A rich, interactive UI to:

- Configure detailed race parameters (laps, track selection, driver profiles).
- Define complex race scenarios with probabilistic rain forecasts and scheduled Safety Car periods.
- Run simulations using different strategies: **Q-learning**, **PPO**, **Head-to-Head**, and **Custom**.
- **Statistical Comparison Mode**: Run batch simulations (e.g., 50 races per strategy) to robustly compare the performance distributions of different agents using summary tables and box plots.

### Advanced Data Visualization (with Plotly)

- Real-time animated lap metrics (tire wear, fuel, traffic).
- Strategic Event Timeline with emoji markers: 🅿️ Pits, 🌧️ Rain, ⚠️ SC, 🚦 VSC.
- Detailed post-race analysis charts (lap time delta, tire usage, track conditions).

### ML-Powered Insights & Reporting

- Integrates a separately trained model (`lap_time_predictor.pkl`) to predict lap times based on race conditions, offering a comparison with the simulation's actual outcomes.
- Generates **downloadable PDF race summary reports**.

---

## 🛠️ Technologies Used

- **Core**: Python 3.10+
  
### Simulation Environment
- `gymnasium`
- `NumPy`
- `Pandas`

### Reinforcement Learning
- Custom Q-learning implementation
- `stable-baselines3[extra]` for PPO
- `torch` (backend for SB3)

### Machine Learning (Lap Time Predictor)
- `scikit-learn` (RandomForestRegressor)
- `joblib`

### User Interface & Visualization
- `Streamlit`
- `Plotly`
- `Matplotlib`
- `Seaborn`

### PDF Generation
- `fpdf2`


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
