# 🏎️ F1 Pit Stop Strategy Simulator
[**Launch Simulator**](https://pit-stop-sim.streamlit.app/)

An advanced, interactive Formula 1 pit stop strategy simulator that uses **Reinforcement Learning** and **Machine Learning** to optimize race strategies. This project demonstrates how AI agents can learn optimal pit stop timing and tire compound selection through dynamic race simulations.

---

## 🎯 What This Project Does
This simulator creates a realistic F1 race environment where AI agents learn to make strategic pit stop decisions by considering:

- **Dynamic tire degradation** with realistic wear patterns for different compounds (Soft, Medium, Hard, Intermediate, Wet)  
- **Weather conditions** including probabilistic rain forecasts and intensity changes  
- **Track incidents** like Safety Cars and Virtual Safety Cars that create strategic opportunities  
- **Traffic effects** and fuel consumption that impact lap times  
- **Team characteristics** and driver profiles that influence decision-making  
- **FIA regulations** including mandatory tire compound usage in dry races  

---

## 🧠 Core Innovation
The project combines **Q-Learning** and **Proximal Policy Optimization (PPO)** agents that learn optimal strategies through thousands of simulated races, then provides interactive tools to analyze and compare their performance against custom strategies.

---

## ✨ Key Features

### 🤖 AI-Powered Strategy Agents
- **Q-Learning Agent**: Custom implementation that discretizes the complex race state space to learn optimal pit strategies.  
- **PPO Agent**: Advanced reinforcement learning using Stable Baselines3 for more sophisticated decision-making.  
- **Adaptive Learning**: Considers tire wear, track conditions, weather, and race incidents when making pit decisions.

### 🏁 Comprehensive Race Simulation
- **Realistic Physics**: Non-linear tire degradation, fuel consumption effects, and track grip variation.  
- **Dynamic Events**: Probabilistic rain, Safety Car deployments, and Virtual Safety Car periods.  
- **Multiple Session Types**: Practice, Qualifying, Race, Full Weekend, and Statistical Comparison modes.  
- **Track Variety**: 9 pre-configured circuits (Monza, Spa, Monaco, etc.) plus custom track creation.

### 📊 Advanced Analytics Dashboard
- **Real-time Visualization**: Animated lap-by-lap metrics showing tire wear, fuel levels, and traffic.  
- **Strategic Timeline**: Event markers showing pit stops, rain, Safety Cars with emoji indicators.  
- **Performance Analysis**: Lap time deltas, tire usage patterns, and efficiency ratings.  
- **Statistical Comparison**: Compare strategy performance distributions in batch simulations.

### 🧠 ML-Powered Insights
- **Lap Time Predictor**: Separate RandomForest model predicts lap times based on race conditions.  
- **Performance Comparison**: Actual vs. predicted lap times to validate simulation accuracy.  
- **Feature Importance**: Understand which factors most influence lap time performance.

---

## 🛠️ Technology Stack

**Core Simulation**
- **Python 3.10+** - Primary development language
- **Gymnasium** - RL environment framework for race simulation  
- **NumPy & Pandas** - Data processing and analysis 

**Machine Learning & AI**
- **Custom Q-Learning**  - Tabular reinforcement learning implementation
- **Stable Baselines3**  - PPO agent training and inference
- **PyTorch** - Backend for deep RL models 
- **Scikit-learn** - Lap time prediction model (RandomForestRegressor)

**User Interface & Visualization**
- **Streamlit** - Interactive web application framework  
- **Plotly** - Advanced interactive charts and animations  
- **Matplotlib & Seaborn**  - Statistical visualization and analysis

**Model Management**
- **Hugging Face Hub** - Centralized model storage and distribution  
- **Joblib** - Model serialization and loading  
- **FPDF2** - Automated race report generation  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or newer  
- Git  
- ~2GB disk space for models and data  

### Quick Setup
1. **Clone the Repository**
```
git clone https://github.com/rembertdesigns/pit-stop-simulator.git
cd pit-stop-simulator
```
2. **Create Virtual Environment** (Recommended)
```
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate
```
3. **Install Dependencies**
```
pip install -r requirements.txt
```

---

### Training Your Models  
⚠️ **Important**: You must train the models yourself — they are not provided.

1. **Generate Initial Data**
```
streamlit run streamlit_app.py
```
Navigate to the app and run 2-3 race simulations to generate initial data in `logs/gym_race_lap_data.csv`.

2. **Train Lap Time Predictor**
```
python train_lap_model.py
```
This creates `models/lap_time_predictor.pkl` for ML insights.

3. **Train Q-Learning Agents**
```
python main.py
```
This trains agents for all team/profile combinations and saves them to `saved_agents/`.

4. **Train PPO Agent**
```
python train_ppo.py
```
This creates `models/ppo_pit_stop.zip` for advanced RL strategies.

### Launch the Simulator
```
streamlit run streamlit_app.py
```
Open your browser to http://localhost:8501 and start simulating!

---

## 📁 Project Structure
```
pit-stop-simulator/
│
├── streamlit_app.py              # 🖥️  Main web application with full UI
│
├── env/
│   └── gym_race_env.py           # 🏁 Custom F1 race environment (Gymnasium)
│
├── rl/
│   └── q_learning_agent.py       # 🧠 Q-Learning agent implementation
│
├── train_ppo.py                  # 🚂 PPO agent training script
├── train_lap_model.py            # 📈 ML lap time predictor training
├── main.py                       # 🎯 Batch Q-Learning agent training
├── ppo_eval.py                   # 📊 PPO agent evaluation script
│
├── models/                       # 🤖 (Auto-downloaded) Trained models
├── saved_agents/                 # 💾 (Auto-generated) Q-Learning agents
├── logs/                         # 📝 (Auto-generated) Simulation data
│
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # 📚 This documentation
```

---

## 🎮 How to Use

### Basic Simulation

1. **Select Strategy:** Choose between Q-Learning, PPO, or Custom pit strategies
2. **Configure Race:** Set laps (20-80), select track, and choose team/driver profile
3. **Add Events:** Configure rain forecasts, Safety Car deployments
4. **Run Simulation:** Watch real-time animated lap metrics and strategic decisions
5. **Analyze Results:** Review lap time deltas, tire usage, and ML predictions

### Advanced Analysis

- **Head-to-Head Mode:** Direct comparison between Q-Learning and PPO agents
- **Statistical Comparison:** Run 10-100 races per strategy to compare performance distributions
- **Full Weekend:** Experience Practice → Qualifying → Race progression
- **Custom Strategies:** Define manual pit stop laps to test specific approaches

### Understanding the Agents
**Q-Learning Agent:**

- Uses tabular learning with discretized state space
- Buckets: (laps, tire_wear, traffic, rain, safety_car, vsc)
- Action space: 0=Stay Out, 1=Pit (with tire selection logic)
- Best for: Consistent, learned patterns based on discrete state combinations

**PPO Agent:**

- Neural network-based continuous learning
- Observation space: 7-dimensional (lap, tire_wear, traffic, fuel, rain, sc, vsc)
- Action space: 6 actions (Stay Out, Pit for Soft/Medium/Hard/Intermediate/Wet)
- Best for: Complex pattern recognition and nuanced decision-making

---

## 🔧 Advanced Configuration

### Custom Track Creation
Set `Track: Custom` in the sidebar to define:

- **Pit Stop Time:** Base time penalty (20-40s)
- **Track Abrasiveness:** Tire wear multiplier (0.5-2.5x)
- **Traffic Penalty:** Time lost in traffic (1.0-10.0s)
- **Base Lap Time:** Ideal dry conditions pace (60-120s)

### Driver Profile Effects

- **Aggressive:** Higher tire wear tolerance (75%), faster pace, +15% tire wear
- **Balanced:** Standard thresholds (65% wear), neutral bonuses
- **Conservative:** Early pit strategy (55% wear), cautious approach, -15% tire wear

### Weather Simulation

- **Probabilistic Forecasts:** Define lap ranges with rain probability percentages
- **Dynamic Intensity:** Rain strength affects lap times and tire wear
- **Tire Strategy:** Wet conditions favor Intermediate/Wet tire compounds

---

## 📊 Key Metrics Explained

- **Total Reward:** Cumulative lap time performance (higher = better overall race time)
- **Pit Efficiency:** Estimated time optimization from pit strategy decisions
- **Tire Wear Patterns:** Degradation curves showing optimal pit windows
- **Traffic Impact:** Effect of other cars on lap time performance
- **FIA Penalties:** Automatic penalties for regulation violations (e.g., not using 2 compounds in dry races)

---

### 🛣️ Roadmap & Future Enhancements

- [ ] **Multi-Agent Racing:** Simulate full grids with competing AI strategies
- [ ] **What-If Analysis:** Replay races with modified decisions to see impact
- [ ] **Advanced Driver Models:** Skill-based performance variation and errors
- [ ] **Real-Time Strategy Updates:** Mid-race strategy adjustments based on conditions
- [ ] **Enhanced ML Models:** Deep learning for more sophisticated lap time prediction
- [ ] **Championship Mode:** Season-long strategy optimization across multiple races
