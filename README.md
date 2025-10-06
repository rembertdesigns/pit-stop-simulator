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



## 🔗 Model Repository
All trained models are hosted on [Hugging Face Hub: **Richard1224/pit-stop-simulator-models**](https://huggingface.co/Richard1224/pit-stop-simulator-models)  


