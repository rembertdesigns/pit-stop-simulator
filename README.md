# 🏎️ F1 Pit Stop Strategy Simulator
[**Launch Simulator**](https://pit-stop-sim.streamlit.app/)

An advanced, interactive Formula 1 pit stop strategy simulator that uses **Reinforcement Learning** and **Machine Learning** to optimize race strategies.  
This project demonstrates how AI agents can learn optimal pit stop timing and tire compound selection through dynamic race simulations.

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
- Python 3.10+  
- Gymnasium  
- NumPy & Pandas  

**Machine Learning & AI**
- Custom Q-Learning (Tabular)  
- Stable Baselines3 (PPO)  
- PyTorch  
- Scikit-learn (RandomForestRegressor)

**User Interface & Visualization**
- Streamlit  
- Plotly  
- Matplotlib & Seaborn

**Model Management**
- Hugging Face Hub  
- Joblib  
- FPDF2  

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
