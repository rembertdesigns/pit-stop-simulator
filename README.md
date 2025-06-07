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
├── streamlit_app.py         # Main Streamlit application: UI and simulation logic
│
├── env/
│   └── gym_race_env.py      # Custom Gymnasium F1 race simulation environment
│
├── rl/
│   └── q_learning_agent.py  # Q-learning agent implementation
│
├── models/                  # (Gitignored) Directory for trained models
│   ├── ppo_pit_stop.zip     
│   └── lap_time_predictor.pkl
│
├── saved_agents/            # (Gitignored) Directory for saved Q-learning agent Q-tables
│   └── e.g., Ferrari_Balanced_q.pkl
│
├── train_ppo.py             # Script to train the PPO agent
├── train_lap_model.py       # Script to train the ML lap time predictor model
├── main.py                  # Script for batch-training Q-Learning agents
├── ppo_eval.py              # Example script to evaluate a trained PPO agent
│
├── logs/
│   └── gym_race_lap_data.csv # Detailed lap-by-lap log data from simulations
│
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```
---

## ⚙️ Simulation Core Details

The `PitStopEnv` provides a **7-dimensional observation space** to the agents:

- **Current Lap**: `[0, total_laps]`
- **Tire Wear**: `[0.0, 100.0]`
- **Traffic Level**: `[0.0, 1.0]`
- **Fuel Weight**: `[0.0, 110.0]` (in kg)
- **Rain Active**: `0` or `1`
- **Safety Car Active**: `0` or `1`
- **VSC Active**: `0` or `1`

### Actions Available to Agents (`Discrete(6)`):

- `0`: Stay Out  
- `1`: Pit for Soft Tires  
- `2`: Pit for Medium Tires  
- `3`: Pit for Hard Tires  
- `4`: Pit for Intermediate Tires  
- `5`: Pit for Wet Tires  

### Reward System

The primary reward is the **negative of the lap time**. Agents aim to maximize this cumulative reward, which is equivalent to **minimizing the total race time**. Bonuses from driver profiles are also factored in.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` for package installation

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/rembertdesigns/pit-stop-simulator.git
cd pit-stop-simulator
```
2. **Create and activate a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```
3. **Install dependencies from `requirements.txt`:**

```bash
pip install -r requirements.txt
```
(Note: Your `requirements.txt` should be updated to include `streamlit`, `pandas`, `plotly`, `fpdf2`, `gymnasium`, `scikit-learn`, `stable-baselines3[extra]`, etc.)

### Running the Simulator

1. **Train Your Models (Required First Step):**

- The trained models are not included in the repository. You must train them yourself first. Follow the sequence below.
- **a. Generate Simulation Data:** Run `streamlit run streamlit_app.py` and execute a few simulations to generate log data in `logs/gym_race_lap_data.csv`.
- **b. Train Lap Time Predictor:** Run `python3 train_lap_model.py`. This creates `models/lap_time_predictor.pkl`.
- **c. Train Q-Learning Agents:** Run `python3 main.py` to batch-train agents for all team/profile combinations. This populates `saved_agents/`.
- **d. Train PPO Agent:** Run `python3 train_ppo.py`. This creates `models/ppo_pit_stop.zip`.

2. **Launch the Streamlit Application:**

Once models are trained, launch the app:
```bash
streamlit run streamlit_app.py
```
Open the URL provided in your terminal (usually `http://localhost:8501`) in your web browser.
