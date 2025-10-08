# 🏎️ F1 Pit Stop Strategy Simulator
[**Launch Simulator**](https://pit-stop-sim.streamlit.app/)

<img 
  width="1536" 
  height="1024" 
  alt="F1 Pit Stop Strategy Simulator - AI-powered Formula 1 race strategy optimization tool with reinforcement learning for tire management and pit timing decisions" 
  title="F1 Pit Stop Strategy Simulator - AI Formula 1 Race Strategy Optimization Tool"
  src="https://github.com/user-attachments/assets/f61b6403-1633-4b13-83ab-ccdff35568dd"
  loading="lazy"
/>

An advanced, interactive Formula 1 pit stop strategy simulator leveraging **Reinforcement Learning** and **Machine Learning** to optimize race strategies through dynamic simulation. This project demonstrates AI-driven decision-making for pit stop timing and tire compound selection in realistic race environments.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Core Innovation](#-core-innovation)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Installation](#-installation)
- [Model Training Pipeline](#-model-training-pipeline)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Performance Metrics](#-performance-metrics)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This simulator creates a high-fidelity F1 race environment where AI agents learn optimal pit stop strategies through reinforcement learning. The system models complex race dynamics including:

- **Race Dynamics Simulation**
  - **Tire Degradation Models:** Non-linear wear patterns with compound-specific coefficients  
    - Soft: High grip, rapid degradation (1.5x base rate)
    - Medium: Balanced performance (1.0x base rate)
    - Hard: Durability focus (0.7x base rate)
    - Intermediate/Wet: Weather-dependent characteristics

- **Environmental Factors**
  - Dynamic weather system with probabilistic rain forecasts
  - Track temperature variations (20-50°C range)
  - Grip factor evolution based on rubber buildup and conditions

- **Race Incidents**
  - Safety Car (SC) deployments with configurable duration
  - Virtual Safety Car (VSC) periods
  - Strategic opportunity windows during caution periods

- **Performance Variables**
  - Traffic modeling with time penalties (1-10s per lap)
  - Fuel consumption effects on lap times (~0.03s per kg)
  - FIA regulation compliance (mandatory compound usage)

  ---

## 🧠 Core Innovation

### Dual-Agent Architecture

The project implements two complementary reinforcement learning approaches:

1. **Q-Learning Agent**
   - **Algorithm:** Tabular Q-Learning with discretized state space
   - **State Space Discretization:**
```python
State = (
    lap_bucket,           # 0-3 (race phase)
    tire_wear_bucket,     # 0-9 (0-100% in 10% increments)
    traffic_bucket,       # 0-2 (low/medium/high)
    rain_status,          # 0-1 (binary)
    safety_car_status,    # 0-1 (binary)
    vsc_status           # 0-1 (binary)
)
```
**Action Space:** 6 discrete actions

- **Action 0:** Stay out (no pit)
- **Actions 1-5:** Pit for specific compound (Soft / Medium / Hard / Intermediate / Wet)

**Learning Parameters:**

- **Learning rate (α):** 0.1
- **Discount factor (γ):** 0.99
- **Epsilon-greedy exploration:** ε = 1.0 → 0.01 (decay over 2000 episodes)

2. **PPO Agent (Proximal Policy Optimization)**
  - Framework: Stable-Baselines3 implementation
  - Observation Space: 7-dimensional continuous vector
```python
obs = [
    current_lap / total_laps,        # Normalized race progress
    tire_wear / 100.0,               # Normalized tire condition
    traffic,                         # Traffic intensity [0, 1]
    fuel_weight / initial_fuel,      # Normalized fuel load
    rain_intensity,                  # Rain strength [0, 1]
    safety_car_active,               # Binary flag
    vsc_active                       # Binary flag
]
```
**Network Architecture**

- Policy Network: MLP with 2 hidden layers (64 neurons each)
- Value Network: Shared feature extraction  
- Activation: Tanh

**Training Hyperparameters**

- Total timesteps: 300,000  
- Batch size: 64  
- Learning rate: 2.5e-4  
- GAE λ: 0.95  
- Clip range: 0.2

---

## ✨ Key Features

### 🤖 AI Strategy Agents

**Q-Learning Agent**
- Fast inference (~0.1ms per decision)  
- Interpretable state-action mappings  
- Optimal for discrete, well-defined scenarios  
- 15 agents (5 teams × 3 driver profiles)

**PPO Agent**
- Continuous learning capability  
- Superior generalization to novel conditions  
- Neural network-based policy approximation  
- Single model handles all team/profile combinations

**Performance Comparison**
- Head-to-Head mode for direct agent comparison  
- Statistical analysis over 100+ race simulations  
- Distribution plots for race time and pit stop counts

### Race Simulation Engine

**Session Types**
1. **Practice:** 3 stints × 3 laps, random tire compounds  
2. **Qualifying:** 3 flying laps, best time recording  
3. **Race:** Full distance with strategic pit stops  
4. **Full Weekend:** Complete P → Q → R progression  
5. **Statistical Comparison:** Batch simulations (10–100 runs)

**Circuit Library:** 9 Pre-Configured Tracks
| Track       | Pit Time | Wear Rate | Traffic | Base Lap |
|--------------|-----------|-----------|----------|-----------|
| Monza        | 28s       | 1.1x      | 3.0s     | 80.0s     |
| Spa          | 32s       | 1.2x      | 4.0s     | 105.0s    |
| Monaco       | 25s       | 1.4x      | 7.5s     | 71.0s     |
| Bahrain      | 30s       | 2.0x      | 5.0s     | 92.0s     |
| Silverstone  | 29s       | 1.8x      | 4.5s     | 88.0s     |

### 📊 Analytics Dashboard
**Real-Time Visualizations** (Plotly-powered)
- Animated lap-by-lap metrics with **0.001–0.5s replay speed**  
- Tire wear progression with **rolling average smoothing**  
- **Traffic intensity heatmaps**  
- **Fuel consumption curves**

**Post-Race Analysis**
- **Lap time delta charts** (vs. first valid lap baseline)  
- **Strategic event timeline** with emoji markers (🅿️ 🌧️ ⚠️ 🚦)  
- **Tire compound usage distribution**  
- **Track temperature and grip factor evolution**

**ML Insights**
- **RandomForest lap time predictions** vs. actual performance  
- **Feature importance rankings**  
- **Prediction error analysis** (RMSE, R²)

### 🧠 Machine Learning Pipeline
**Lap Time Predictor**
- **Model:** `RandomForestRegressor (150 trees)`  
- **Features:** 14 inputs (7 numeric + 5 tire compound dummies + 2 boolean)  
- **Training data:** Aggregated from simulation logs  
- **Typical performance:** R² > 0.85, RMSE < 2.0s

---

## 🛠️ Technical Architecture

### Technology Stack
<img width="421" height="433" alt="Screenshot 2025-10-06 at 2 39 09 PM" src="https://github.com/user-attachments/assets/24571e4c-2d1b-4934-bcfc-1597e2fb8fa6" />

### Core Dependencies

| Package           | Version | Purpose                        |
|--------------------|----------|--------------------------------|
| **Python**         | 3.10+    | Base runtime                   |
| **Streamlit**      | 1.28+    | Web application framework      |
| **Gymnasium**      | 0.29+    | RL environment standard        |
| **Stable-Baselines3** | 2.1+ | PPO implementation             |
| **PyTorch**        | 2.0+     | Deep learning backend          |
| **Scikit-learn**   | 1.3+     | ML predictor model             |
| **Plotly**         | 5.17+    | Interactive visualizations     |
| **NumPy**          | 1.24+    | Numerical computing            |
| **Pandas**         | 2.0+     | Data manipulation              |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or newer  
- Git  
- ~2GB disk space for models and dependencies  
- 4GB+ RAM recommended for PPO training  

### Quick Setup
1. **Clone Repository**
```bash
git clone https://github.com/rembertdesigns/pit-stop-simulator.git
cd pit-stop-simulator
```
2. **Create Virtual Environment** (Recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. **Verify Installation**
```bash
python -c "import streamlit; import gymnasium; import stable_baselines3; print('✅ All dependencies installed')"
```
---

## 🎓 Model Training Pipeline

### ⚠️ Important Notice
Pre-trained models are **not included** in the repository. You must train them yourself using the provided scripts.

### Training Workflow
```bash
1. Generate Data → 2. Train ML Predictor → 3. Train Q-Agents → 4. Train PPO Agent
```
### Step 1: Generate Initial Data

Run 2-3 race simulations to create training data:
```bash
streamlit run streamlit_app.py
```
**Action Items**:
- Navigate to the app (http://localhost:8501)
- Configure race settings (50+ laps recommended)
- Run simulations with varied conditions (rain, SC, different tracks)
- Verify `logs/gym_race_lap_data.csv` contains data (should have 100+ rows)

### Step 2: Train Lap Time Predictor
```bash
python train_lap_model.py
```
**What This Does**:
- Loads data from `logs/gym_race_lap_data.csv`
- Preprocesses features (one-hot encoding, normalization)
- Trains RandomForestRegressor
- Saves model to `models/lap_time_predictor.pkl`
- Prints evaluation metrics (RMSE, R², feature importances)

**Expected Output**:
```bash
Training set size: 800 samples, Test set size: 200 samples
Training RandomForestRegressor model with 14 features...
Model training complete.

--- Model Evaluation on Test Set ---
  Root Mean Squared Error (RMSE): 1.847
  R-squared (R2 Score):           0.891

✅ Model retrained and saved to: models/lap_time_predictor.pkl
```
### Step 3: Train Q-Learning Agents
```
python main.py
```
**Configuration** (in `main.py`):
```python
TEAMS_TO_TRAIN = ["Ferrari", "Red Bull", "Mercedes", "McLaren", "Aston Martin"]
PROFILES_TO_TRAIN = ["Aggressive", "Balanced", "Conservative"]
TRAINING_EPISODES = 2000
```
**What This Does**:
- Creates 15 agents (5 teams × 3 profiles)
- Trains each agent for 2000 episodes (~30 min on modern CPU)
- Saves agents to `saved_agents/{Team}_{Profile}_q.pkl`
- Generates training plots in `training_figures/`

**Expected Output** (per agent):
```bash
--- Initializing training for: Ferrari - Aggressive ---
Training Ferrari Aggressive: 100%|████████| 2000/2000 [05:42<00:00, 5.84it/s]
--- Training for Ferrari - Aggressive complete ---
✅ Agent successfully saved to: saved_agents/Ferrari_Aggressive_q.pkl
```
### Step 4: Train PPO Agent
```bash
python train_ppo.py
```
**What This Does**:
- Creates vectorized training environment
- Trains PPO agent for 300,000 timesteps (~2-3 hours on CPU, ~30 min on GPU)
- Saves checkpoints every 10,000 steps
- Evaluates and saves best model
- Final model saved to `models/ppo_pit_stop.zip`

**Expected Output**:
```bash
Creating training environment...
PPO Model Created. Observation Space: Box(7,), Action Space: Discrete(6)
Starting PPO training for 300000 timesteps...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.0     |
|    ep_rew_mean     | -5247.32 |
| time/              |          |
|    fps             | 1247     |
|    iterations      | 146      |
|    time_elapsed    | 240      |
|    total_timesteps | 299008   |
...
✅ Training complete. Final model saved to: models/ppo_pit_stop.zip
```
### Verifying Training Success

After training, verify all models exist:
```bash
ls -lh models/
# Should show:
# lap_time_predictor.pkl
# ppo_pit_stop.zip

ls -lh saved_agents/
# Should show 15 .pkl files for Q-agents
```

---

## 📖 Usage Guide

## Launch Application
```bash
bashstreamlit run streamlit_app.py
```
Access at: `http://localhost:8501`

### Basic Simulation Workflow

1. **Select Strategy** (Sidebar)
   - Q-Learning: Uses trained Q-table agent
   - PPO: Uses neural network policy
   - Custom: Manual pit lap selection

2. **Configure Race Parameters**
   - Total Laps: 20-80 (default: 58)
   - Track: Select from 9 circuits or define custom
   - Team & Driver Profile: Affects pit thresholds and tire wear

3. **Add Race Events** (Optional)
   - Rain Forecast: Define probability windows
   - Safety Car: Select specific laps
   - Initial Tire: Soft/Medium/Hard

4. **Run Simulation**
   - Click "▶️ Start Simulation"
   - Watch animated lap metrics
   - Review post-race analytics

5. **Analyze Results**
   - Lap time deltas
   - Strategic event timeline
   - ML predictions vs. actual
   - Download PDF report

### Advanced Features

#### Head-to-Head Mode
Compare Q-Learning vs. PPO directly:
- Session Type: "Head-to-Head"
- Side-by-side race visualization
- Comparative summary table

#### Statistical Comparison
Robust strategy evaluation:
- Session Type: "Statistical Comparison"
- Select 2-3 strategies
- Configure 10-100 runs per strategy
- View distribution plots (box plots, histograms)

#### Full Weekend Mode
Experience P → Q → R progression:
- Session Type: "Full Weekend"
- Simulates Practice (9 laps), Qualifying (3 laps), Race (full distance)
- Carries strategic insights between sessions

---

## 📁 Project Structure
```bash
pit-stop-simulator/
│
├── streamlit_app.py              # 🖥️  Main web application (3500+ lines)
│   ├── Model downloader (Hugging Face Hub)
│   ├── Simulation orchestration
│   ├── Plotly visualizations
│   └── PDF report generation
│
├── env/
│   └── gym_race_env.py           # 🏁 Gymnasium-compliant F1 environment
│       ├── State space definition
│       ├── Reward function (lap time + penalties)
│       ├── Weather/SC/VSC logic
│       └── Tire/fuel degradation models
│
├── rl/
│   └── q_learning_agent.py       # 🧠 Tabular Q-Learning implementation
│       ├── State discretization (6D → buckets)
│       ├── Epsilon-greedy exploration
│       └── Q-table updates (Bellman equation)
│
├── train_ppo.py                  # 🚂 PPO agent training script
│   ├── Vectorized environment setup
│   ├── Hyperparameter configuration
│   ├── Checkpoint/eval callbacks
│   └── TensorBoard logging
│
├── train_lap_model.py            # 📈 ML lap time predictor training
│   ├── CSV data loading/preprocessing
│   ├── RandomForest training
│   ├── Feature engineering (one-hot encoding)
│   └── Model evaluation (RMSE, R²)
│
├── main.py                       # 🎯 Batch Q-Learning agent training
│   ├── Multi-agent training loop (15 agents)
│   ├── Progress tracking (tqdm)
│   ├── Training visualization (rewards, heatmaps)
│   └── Agent persistence (pickle)
│
├── ppo_eval.py                   # 📊 PPO agent evaluation script
│   ├── Deterministic policy rollouts
│   ├── Detailed lap-by-lap logging
│   └── Episode statistics
│
├── models/                       # 🤖 Trained models (auto-downloaded from HF Hub)
│   ├── ppo_pit_stop.zip          # PPO neural network policy
│   └── lap_time_predictor.pkl    # RandomForest regressor
│
├── saved_agents/                 # 💾 Q-Learning agents (15 .pkl files)
│   ├── Ferrari_Aggressive_q.pkl
│   ├── Mercedes_Balanced_q.pkl
│   └── ... (13 more combinations)
│
├── logs/                         # 📝 Simulation data (auto-generated)
│   └── gym_race_lap_data.csv     # Lap-by-lap race logs
│
├── training_figures/             # 📊 Training visualizations (auto-generated)
│   ├── rewards_{Team}_{Profile}.png
│   └── heatmap_{Team}_{Profile}.png
│
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📚 This documentation
└── LICENSE                       # ⚖️  MIT License
```

---

## ⚙️ Configuration

### Environment Parameters
**Track Configuration** (`PitStopEnv` initialization):
```python
env = PitStopEnv(
    total_laps=58,                  # Race distance
    base_lap_time_seconds=90.0,     # Ideal dry lap time
    pit_time=28,                    # Pit stop time loss
    tire_wear_rate_config=1.1,      # Track abrasiveness multiplier
    traffic_penalty_config=3.0      # Time lost per traffic unit
)
```
### Driver Profile Effects

Profiles modify agent behavior and physics:

| Profile | Pit Threshold | Tire Wear Multiplier | Overtake Bonus |
|---------|---------------|----------------------|----------------|
| **Aggressive** | 75% wear | 1.15× (faster degradation) | -0.2s (time gain) |
| **Balanced** | 65% wear | 1.0× (baseline) | 0.0s (neutral) |
| **Conservative** | 55% wear | 0.85× (slower degradation) | +0.1s (time loss) |

### Weather System

**Rain Forecast Configuration**:
```python
rain_forecast_ranges = [
    {
        "start": 15,           # Lap window start
        "end": 25,             # Lap window end
        "probability": 0.7,    # 70% chance of rain
        "intensity": 0.6       # Rain strength (0.0-1.0)
    }
]
```
**Rain Effects:**
- Lap time penalty: +5-15s (intensity-dependent)
- Grip reduction: -30% to -60%
- Forces Intermediate/Wet tire compound

---

## 📊 Performance Metrics

### Agent Evaluation Metrics
**Total Reward:** Cumulative race performance
```bash
Total Reward = Σ(lap_rewards) - penalties
where lap_reward = -(lap_time) + bonuses
```
**Pit Efficiency Rating:** Time optimization estimate
```bash
Pit Efficiency = 100 × (1 - (pit_stops × pit_time) / total_race_time)
```
**FIA Penalties**:
- Missing 2-compound rule (dry races): -20 units
- Unsafe release: -10 units (future feature)

### Typical Performance Ranges

| Metric | Q-Learning | PPO | Custom |
|--------|------------|-----|--------|
| Total Reward | -5500 to -4800 | -5300 to -4700 | -6000 to -5000 |
| Avg Pit Stops | 1-3 | 1-2 | User-defined |
| Pit Efficiency | 85-92% | 88-95% | 70-95% |

### ML Model Accuracy

**Lap Time Predictor**:
- RMSE: 1.5-2.5 seconds
- R²: 0.85-0.92
- Feature Importance: tire_wear (35%), lap (18%), fuel_weight (14%)

---

## 🛣️ Roadmap

### Phase 1: Multi-Agent Racing (Q3 2025)
- [ ] Implement 20-car grid simulation
- [ ] Inter-agent competition with overtaking logic
- [ ] Position-based reward shaping
- [ ] Qualifying-based grid positions

### Phase 2: Advanced Strategy (Q4 2025)
- [ ] Dynamic pit strategy updates (mid-race replanning)
- [ ] "What-If" scenario analysis tool
- [ ] Tire compound optimizer (pre-race selection)
- [ ] Undercut/overcut detection system

### Phase 3: Enhanced Realism (Q1 2026)
- [ ] Red flag handling and race restarts
- [ ] Driver skill variation (consistency, errors)
- [ ] Team radio communication simulation
- [ ] Pit crew performance modeling

### Phase 4: Deep Learning (Q2 2026)
- [ ] Transformer-based strategy predictor
- [ ] LSTM for lap time forecasting
- [ ] Convolutional networks for track layout analysis
- [ ] Multi-agent reinforcement learning (MARL)

### Phase 5: Championship Mode (Q3 2026)
- [ ] Full season simulation (23 races)
- [ ] Points system and standings
- [ ] Car development progression
- [ ] Strategic resource allocation (tire allocation)

---

## 🔗 External Resources

### Model Repository
All trained models hosted on [Hugging Face Hub](https://huggingface.co/Richard1224/pit-stop-simulator-models)

**Automatic Download**: Models are fetched on first app launch via `download_models_from_hf()` function.

**Manual Download** (if needed):
```bash
huggingface-cli download Richard1224/pit-stop-simulator-models --local-dir ./
```
### Documentation
- **Stable-Baselines3**: [RL algorithms documentation](https://stable-baselines3.readthedocs.io/)
- **Gymnasium**: [Environment API reference](https://gymnasium.farama.org/)
- **Streamlit**: [Component library](https://docs.streamlit.io/)

---

## 🙌 Contributing

Contributions welcome! Priority areas:

### High-Impact Improvements
1. **Real F1 Data Integration**: Parse actual race telemetry (F1 API, FastF1 library)
2. **Additional RL Algorithms**: A3C, SAC, TD3 implementations
3. **Performance Optimization**: Numba JIT compilation, parallel simulation
4. **Track Expansion**: Add 2024 F1 calendar circuits with accurate characteristics

### Development Setup
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/pit-stop-simulator.git
cd pit-stop-simulator

# Create feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (if implemented)
pytest tests/

# Submit PR with detailed description
```
### Code Standards
- Follow PEP 8 style guide
- Add docstrings to new functions
- Include type hints for function signatures
- Update README if adding major features

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.
```bash
MIT License - Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🏆 Acknowledgments

Built with passion for Formula 1 and powered by:

- **[Streamlit](https://streamlit.io/)** — Rapid data app development framework
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — Production-ready RL implementations
- **[Gymnasium](https://gymnasium.farama.org/)** — Industry-standard RL environment API
- **[Plotly](https://plotly.com/python/)** — Interactive visualization library
- **[Scikit-learn](https://scikit-learn.org/)** — Machine learning fundamentals
- **[Hugging Face Hub](https://huggingface.co/)** — Model hosting and distribution

### Inspiration
This project honors the strategic complexity and split-second decisions that define Formula 1 as the pinnacle of motorsport. Every pit stop, tire choice, and weather call can mean the difference between victory and defeat.

---

<div align="center">

**[⬆ Back to Table of Contents](#-table-of-contents)**

Made with ❤️ for F1 fans and ML enthusiasts

[Report Bug](https://github.com/rembertdesigns/pit-stop-simulator/issues) · [Request Feature](https://github.com/rembertdesigns/pit-stop-simulator/issues) · [Documentation](https://github.com/rembertdesigns/pit-stop-simulator/wiki)

</div>


