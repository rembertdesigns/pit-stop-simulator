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
<img 
  width="1584" 
  height="396" 
  alt="F1 Pit Stop Strategy Simulator Overview - AI-powered Formula 1 race strategy optimization with reinforcement learning for tire management and pit timing" 
  title="F1 Race Strategy Simulator - Machine Learning Pit Stop Optimization Tool"
  src="https://github.com/user-attachments/assets/6db0dd80-5352-425c-8d94-489ff2dd5ec4"
  loading="lazy"
/>

This **AI-powered F1 pit stop simulator** creates a high-fidelity Formula 1 race environment where **reinforcement learning agents** optimize pit stop timing and tire strategy decisions. The system models realistic race dynamics using **machine learning** to predict optimal compound selection and pit windows.

### 🏎️ Race Dynamics Simulation

**Tire Degradation Models** - Compound-specific wear algorithms:
- **Soft Compound:** Maximum grip, aggressive degradation (1.5× base wear rate)
- **Medium Compound:** Balanced performance and longevity (1.0× base wear rate)  
- **Hard Compound:** Extended stint capability (0.7× base wear rate)
- **Wet/Intermediate:** Dynamic performance based on track conditions

### 🌦️ Environmental Simulation

**Weather & Track Conditions:**
- Probabilistic rain forecast system with intensity modeling
- Track temperature simulation (20-50°C operational range)
- Dynamic grip evolution (rubber buildup, track cleaning effects)

### ⚠️ Race Incident Modeling

**Safety Protocols:**
- **Safety Car (SC)** deployments with configurable duration
- **Virtual Safety Car (VSC)** period simulation
- Strategic pit window optimization during caution phases

### 📊 Performance Variables

**Realistic Race Physics:**
- Traffic simulation with time penalties (1-10 seconds per lap)
- Fuel load effects on lap times (~0.03s per kilogram)
- FIA technical regulation compliance (mandatory tire compound rules)

**[⬆ Back to Table of Contents](#-table-of-contents)**

  ---

## 🧠 Core Innovation
<img 
  width="1584" 
  height="396" 
  alt="Core Innovation - Dual-Agent Reinforcement Learning Architecture for F1 Pit Stop Strategy Optimization using Q-Learning and PPO algorithms" 
  title="AI Racing Strategy - Q-Learning vs PPO Agent Comparison for Formula 1 Pit Stop Optimization"
  src="https://github.com/user-attachments/assets/2ef46d01-cd65-4bc9-ae1b-db946d70cc77"
  loading="lazy"
/>

### Dual-Agent Architecture

This **F1 strategy simulator** implements two complementary **AI racing agents** using cutting-edge **reinforcement learning algorithms** for optimal pit stop decision-making.

### 1️⃣ Q-Learning Agent - Tabular Reinforcement Learning

#### State Space Representation (6-Dimensional Discretization)
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
#### Action Space - 6 Discrete Pit Stop Strategies

- **Action 0:** Continue racing (no pit stop)
- **Actions 1-5:** Execute pit stop with tire selection:
  - Soft compound (maximum grip)
  - Medium compound (balanced)
  - Hard compound (durability)
  - Intermediate (light rain)
  - Wet (heavy rain)

#### Q-Learning Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Learning Rate (α)** | 0.1 | Q-value update step size |
| **Discount Factor (γ)** | 0.99 | Future reward importance |
| **Exploration Rate (ε)** | 1.0 → 0.01 | Decay over 2000 episodes |

### 2️⃣ PPO Agent - Deep Reinforcement Learning

**Framework:** Stable-Baselines3 Proximal Policy Optimization (PyTorch backend)

#### Continuous Observation Space (7 Features)
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
#### Neural Network Architecture

**Policy Network (Actor):**
- Multi-Layer Perceptron (MLP)
- 2 hidden layers × 64 neurons
- Tanh activation function
- Outputs: Probability distribution over 6 actions

**Value Network (Critic):**
- Shared feature extraction with policy network
- Estimates state value function V(s)

#### PPO Training Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| **Total Timesteps** | 300,000 | Training iterations |
| **Batch Size** | 64 | Samples per gradient update |
| **Learning Rate** | 2.5e-4 | Adam optimizer step size |
| **GAE Lambda (λ)** | 0.95 | Advantage estimation decay |
| **Clip Range** | 0.2 | Policy update constraint (PPO) |

### 🆚 Agent Comparison

| Feature | Q-Learning | PPO |
|---------|------------|-----|
| **State Space** | Discrete (6D buckets) | Continuous (7D vector) |
| **Policy Type** | Tabular lookup | Neural network |
| **Training Speed** | Fast (minutes) | Moderate (hours) |
| **Generalization** | Limited to seen states | Excellent extrapolation |
| **Best For** | Well-defined scenarios | Complex, variable conditions |

**[⬆ Back to Table of Contents](#-table-of-contents)**

---

## ✨ Key Features
<img 
  width="1584" 
  height="396" 
  alt="Key Features - AI-Powered F1 Race Simulation with Q-Learning and PPO Agents, Real-Time Analytics, and Machine Learning Predictions" 
  title="F1 Pit Stop Simulator Features - Reinforcement Learning Racing Strategy Tool with Advanced Analytics"
  src="https://github.com/user-attachments/assets/f1da227e-81e6-4007-a92d-c49f72ec078d"
  loading="lazy"
/>

### 🤖 AI Strategy Agents - Dual Reinforcement Learning System

#### Q-Learning Agent - Fast Tabular RL

- **Lightning-fast inference:** ~0.1ms per pit stop decision
- **Interpretable AI:** Clear state-action mapping for strategy analysis
- **Specialized agents:** 15 pre-trained models (5 F1 teams × 3 driver profiles)
- **Optimal use case:** Discrete, well-defined race scenarios

#### PPO Agent - Advanced Deep Learning

- **Continuous learning:** Neural network adapts to novel race conditions
- **Superior generalization:** Handles unseen weather patterns and traffic scenarios
- **Universal model:** Single agent adapts to all team/driver combinations
- **State-of-the-art:** Proximal Policy Optimization (Stable-Baselines3)

#### Performance Comparison Tools

- **Head-to-Head mode:** Direct Q-Learning vs PPO agent benchmarking
- **Statistical rigor:** Analysis over 100+ race simulations per strategy
- **Visual analytics:** Distribution plots for race times and pit stop frequency

### 🏁 Race Simulation Engine - Complete F1 Weekend Experience

#### Session Types - Full Motorsport Simulation

1. **Practice Session:** 3 stints × 3 laps with randomized tire compound testing
2. **Qualifying Session:** 3 flying laps, best-time recording (Q1/Q2/Q3 format)
3. **Race Session:** Full distance (20-80 laps) with strategic pit stop windows
4. **Full Weekend Mode:** Complete Practice → Qualifying → Race progression
5. **Statistical Comparison:** Batch simulations (10-100 runs) for strategy validation

#### Circuit Library - 9 Authentic F1 Tracks

Realistic track characteristics with accurate pit times and tire wear rates:

| Track | Pit Loss | Tire Wear | Traffic Penalty | Base Lap Time |
|-------|----------|-----------|-----------------|---------------|
| **Monza** (Italy) | 28s | 1.1× | 3.0s | 80.0s |
| **Spa-Francorchamps** (Belgium) | 32s | 1.2× | 4.0s | 105.0s |
| **Monaco** (Monte Carlo) | 25s | 1.4× | 7.5s | 71.0s |
| **Bahrain** (Sakhir) | 30s | 2.0× | 5.0s | 92.0s |
| **Silverstone** (UK) | 29s | 1.8× | 4.5s | 88.0s |

*Additional tracks: Austin (COTA), Suzuka, Singapore, Interlagos*

### 📊 Analytics Dashboard - Professional Race Telemetry

#### Real-Time Visualizations (Plotly Interactive Charts)

- **Animated lap-by-lap metrics:** Adjustable replay speed (0.001-0.5s intervals)
- **Tire degradation tracking:** Rolling average smoothing for trend analysis
- **Traffic intensity heatmaps:** Visual representation of congestion patterns
- **Fuel consumption curves:** Weight-adjusted performance modeling

#### Post-Race Analysis Tools

- **Lap time delta charts:** Performance vs. baseline (first valid lap reference)
- **Strategic event timeline:** Visual race narrative with emoji markers:
  - 🅿️ Pit stops
  - 🌧️ Weather changes
  - ⚠️ Safety Car deployments
  - 🚦 VSC periods
- **Tire compound distribution:** Usage statistics across race distance
- **Environmental tracking:** Track temperature and grip factor evolution

#### ML Insights - Predictive Analytics

- **RandomForest predictions:** AI-forecasted vs. actual lap times comparison
- **Feature importance rankings:** Identify key performance drivers
- **Error analysis:** RMSE and R² model performance metrics

### 🧠 Machine Learning Pipeline - Predictive Lap Time Model

#### Lap Time Predictor Specifications

**Model Architecture:**
- **Algorithm:** RandomForestRegressor with 150 decision trees
- **Input features:** 14-dimensional feature vector
  - 7 numeric features (tire wear, fuel load, traffic, etc.)
  - 5 tire compound dummies (one-hot encoded)
  - 2 boolean flags (weather, safety car status)

**Training Configuration:**
- **Data source:** Aggregated simulation logs from race history
- **Performance benchmarks:**
  - R² Score: >0.85 (85%+ variance explained)
  - RMSE: <2.0 seconds (prediction accuracy)

**Use Cases:**
- Pre-race strategy optimization
- In-race lap time forecasting
- Post-race performance analysis

**[⬆ Back to Table of Contents](#-table-of-contents)**

---

## 🛠️ Technical Architecture
<img 
  width="1584" 
  height="396" 
  alt="Technical Architecture - F1 Pit Stop Simulator Technology Stack with Python, Streamlit, PyTorch, and Stable-Baselines3 for AI Racing Strategy" 
  title="F1 Simulator Tech Stack - Reinforcement Learning Architecture with Deep Learning and Data Science Libraries"
  src="https://github.com/user-attachments/assets/069e75d9-7424-4448-9ddf-98865422614d"
  loading="lazy"
/>

### Technology Stack - Production-Grade AI Racing Platform
<img 
  width="421" 
  height="433" 
  alt="F1 Pit Stop Simulator Technology Stack Diagram showing Python ecosystem with Streamlit web framework, Gymnasium RL environment, Stable-Baselines3 PPO agents, PyTorch neural networks, Scikit-learn ML models, and Plotly data visualization" 
  title="Technology Stack Architecture - Python AI Racing Simulator Components"
  src="https://github.com/user-attachments/assets/24571e4c-2d1b-4934-bcfc-1597e2fb8fa6"
  loading="lazy"
/>

### Core Dependencies - Enterprise Python Stack

**Foundation & Framework Layer**

| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| **Python** | 3.10+ | Base runtime environment | Type hints, pattern matching, async support |
| **Streamlit** | 1.28+ | Interactive web application framework | Real-time data visualization, responsive UI |

**Reinforcement Learning Infrastructure**

| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| **Gymnasium** | 0.29+ | OpenAI Gym successor for RL environments | Standardized API, action/observation spaces |
| **Stable-Baselines3** | 2.1+ | State-of-the-art RL algorithms | PPO, DQN, SAC implementations |
| **PyTorch** | 2.0+ | Deep learning backend | GPU acceleration, automatic differentiation |

**Machine Learning & Analytics**

| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| **Scikit-learn** | 1.3+ | Classical ML algorithms | RandomForest, preprocessing pipelines |
| **Plotly** | 5.17+ | Interactive data visualization | Animated charts, 3D plots, responsive design |
| **NumPy** | 1.24+ | Numerical computing library | Array operations, linear algebra, statistics |
| **Pandas** | 2.0+ | Data manipulation and analysis | DataFrame operations, CSV handling, time series |

### Architecture Highlights

**🚀 Performance Optimizations:**
- PyTorch GPU acceleration for neural network training
- Vectorized NumPy operations for simulation speed
- Streamlit caching for responsive user experience

**🔧 Development Stack:**
- Python 3.10+ for modern language features
- Type hints for code quality and IDE support
- Modular architecture for easy testing and maintenance

**📊 Data Pipeline:**
1. **Simulation Engine** → Gymnasium environment
2. **RL Training** → Stable-Baselines3 PPO/Q-Learning
3. **ML Prediction** → Scikit-learn RandomForest
4. **Visualization** → Plotly interactive charts
5. **Web Interface** → Streamlit dashboard

**[⬆ Back to Table of Contents](#-table-of-contents)**

---

## 🚀 Installation
<img 
  width="1584" 
  height="396" 
  alt="Installation Guide - Step-by-Step Setup for F1 Pit Stop Simulator with Python 3.10+, Git, and Machine Learning Dependencies" 
  title="F1 Simulator Installation - Quick Setup Guide for Reinforcement Learning Racing Strategy Tool"
  src="https://github.com/user-attachments/assets/e16bd88f-f5cb-493a-8692-6f6b3904f90f"
  loading="lazy"
/>

### Prerequisites - System Requirements

Before installing the F1 Pit Stop Simulator, ensure your system meets these requirements:

**Software Requirements:**
- **Python 3.10 or newer** (3.11+ recommended for performance)
- **Git** version control system
- **pip** package manager (included with Python)

**Hardware Requirements:**
- **Storage:** ~2GB disk space (models + dependencies)
- **RAM:** 4GB+ recommended for PPO training (8GB+ optimal)
- **GPU:** Optional (CUDA-compatible for faster training)

### Quick Setup - 5-Minute Installation

#### Step 1: Clone Repository from GitHub

Download the F1 simulator source code:
```bash
git clone https://github.com/rembertdesigns/pit-stop-simulator.git
cd pit-stop-simulator
```
**Alternative:** Download ZIP from [GitHub releases](https://github.com/rembertdesigns/pit-stop-simulator/releases)

#### Step 2: Create Virtual Environment (Recommended)

Isolate project dependencies using Python's venv:

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Why virtual environments?** Prevents dependency conflicts and ensures reproducibility.

#### Step 3: Install Python Dependencies

Install all required packages via pip:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**What gets installed:**

- Streamlit (web framework)
- Stable-Baselines3 (RL algorithms)
- PyTorch (deep learning)
- Gymnasium (RL environments)
- Plotly, NumPy, Pandas (analytics)

**Installation time:** ~2-5 minutes depending on internet speed

#### Step 4: Verify Installation

Test that all critical dependencies are installed correctly:
```bash
python -c "import streamlit; import gymnasium; import stable_baselines3; print('✅ All dependencies installed')"
```
**Expected output:**
```bash
✅ All dependencies installed
```
### Next Steps

After successful installation:

1. **Launch the simulator:** `streamlit run streamlit_app.py`
2. **Train models:** Follow the [Model Training Pipeline](#-model-training-pipeline) guide
3. **Run your first race:** See the [Usage Guide](#-usage-guide)

### Troubleshooting Common Issues

**Issue:** `ModuleNotFoundError: No module named 'torch'`  
**Solution:** Reinstall PyTorch: `pip install torch --upgrade`

**Issue:** Installation fails on Windows  
**Solution:** Install Microsoft Visual C++ Build Tools

**Issue:** Out of memory during training  
**Solution:** Reduce batch size in `train_ppo.py` or use Q-Learning agents

**Need help?** [Open an issue on GitHub](https://github.com/rembertdesigns/pit-stop-simulator/issues)

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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

**[⬆ Back to Table of Contents](#-table-of-contents)**

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


