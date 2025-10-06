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




## 🔗 Model Repository
All trained models are hosted on [Hugging Face Hub: **Richard1224/pit-stop-simulator-models**](https://huggingface.co/Richard1224/pit-stop-simulator-models)  


