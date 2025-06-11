# --- Core Streamlit and Data Science Imports ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import pickle
import joblib
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import plotly.express as px
import plotly.graph_objects as go
import random

# --- Hugging Face Model Downloader Import ---
from huggingface_hub import hf_hub_download

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")

# --- HUGGING FACE MODEL DOWNLOADER ---
# This section runs once at the beginning to ensure all models are available.

@st.cache_resource # Cache the download so it doesn't run every time
def download_models_from_hf():
    """
    Downloads all necessary model files from a Hugging Face Hub repo.
    """
    # Your Hugging Face repository ID
    HF_REPO_ID = "Richard1224/pit-stop-simulator-models"

    # A dictionary of remote filenames and their local destination paths
    files_to_download = {
        # PPO Model and data
        "ppo_pit_stop.zip": "models/ppo_pit_stop.zip",
        "ppo_rewards.npy": "data/ppo_rewards.npy",

        # ML Predictor
        "lap_time_predictor.pkl": "models/lap_time_predictor.pkl",

        # Q-Agents
        "Aston Martin_Aggressive_q.pkl": "saved_agents/Aston Martin_Aggressive_q.pkl",
        "Aston Martin_Balanced_q.pkl": "saved_agents/Aston Martin_Balanced_q.pkl",
        "Aston Martin_Conservative_q.pkl": "saved_agents/Aston Martin_Conservative_q.pkl",
        "Ferrari_Aggressive_q.pkl": "saved_agents/Ferrari_Aggressive_q.pkl",
        "Ferrari_Balanced_q.pkl": "saved_agents/Ferrari_Balanced_q.pkl",
        "Ferrari_Conservative_q.pkl": "saved_agents/Ferrari_Conservative_q.pkl",
        "McLaren_Aggressive_q.pkl": "saved_agents/McLaren_Aggressive_q.pkl",
        "McLaren_Balanced_q.pkl": "saved_agents/McLaren_Balanced_q.pkl",
        "McLaren_Conservative_q.pkl": "saved_agents/McLaren_Conservative_q.pkl",
        "Mercedes_Aggressive_q.pkl": "saved_agents/Mercedes_Aggressive_q.pkl",
        "Mercedes_Balanced_q.pkl": "saved_agents/Mercedes_Balanced_q.pkl",
        "Mercedes_Conservative_q.pkl": "saved_agents/Mercedes_Conservative_q.pkl",
        "Red Bull_Aggressive_q.pkl": "saved_agents/Red Bull_Aggressive_q.pkl",
        "Red Bull_Balanced_q.pkl": "saved_agents/Red Bull_Balanced_q.pkl",
        "Red Bull_Conservative_q.pkl": "saved_agents/Red Bull_Conservative_q.pkl",
    }

    # Create local directories
    local_dirs = set(os.path.dirname(path) for path in files_to_download.values())
    for d in local_dirs:
        os.makedirs(d, exist_ok=True)

    # Download each file
    st.info("Downloading simulation models. This may take a moment on first startup...")
    progress_bar = st.progress(0)
    total_files = len(files_to_download)

    for i, (remote_filename, local_path) in enumerate(files_to_download.items()):
        if not os.path.exists(local_path):
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=remote_filename,
                    local_dir=os.path.dirname(local_path),
                    local_dir_use_symlinks=False
                )
                downloaded_file = os.path.join(os.path.dirname(local_path), os.path.basename(remote_filename))
                if downloaded_file != local_path:
                     os.rename(downloaded_file, local_path)
            except Exception as e:
                st.error(f"Error downloading {remote_filename}: {e}")
                return False # Stop if a model fails to download
        
        progress_bar.progress((i + 1) / total_files)

    st.success("All models downloaded successfully!")
    time.sleep(1) # Give user a moment to see the success message
    progress_bar.empty()
    return True

# Run the downloader at the start of the app
models_ready = download_models_from_hf()

# THE REST OF YOUR APPLICATION RUNS ONLY IF MODELS ARE READY
if models_ready:
    # --- Custom Modules (should be loaded after models are confirmed to exist) ---
    from env.gym_race_env import PitStopEnv
    from rl.q_learning_agent import QAgent
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

# --- Initialize Session State Variables ---
if "weekend_phase" not in st.session_state:
    st.session_state.weekend_phase = "Practice"

st.title("🏎️ F1 Pit Stop Strategy Simulator")
st.markdown("An interactive tool to simulate and analyze F1 race strategies using Reinforcement Learning agents.")
st.divider()

# --- Define Dictionaries for Teams and Driver Profiles (BEFORE Sidebar) ---
teams = {
    "Ferrari": {"color": "#DC0000", "logo": "🐎 Leclerc"}, 
    "Red Bull": {"color": "#1E41FF", "logo": "🐂 Verstappen"},
    "Mercedes": {"color": "#00D2BE", "logo": "⭐ Hamilton"}, 
    "McLaren": {"color": "#FF8700", "logo": "🟠 Norris"},
    "Aston Martin": {"color": "#006F62", "logo": "💚 Alonso"},
}

driver_profiles = {
    "Aggressive": {"pit_threshold": 75, "overtake_bonus": -0.2, "tire_wear_impact": 1.15}, 
    "Balanced": {"pit_threshold": 65, "overtake_bonus": 0.0, "tire_wear_impact": 1.0},
    "Conservative": {"pit_threshold": 55, "overtake_bonus": 0.1, "tire_wear_impact": 0.85}
}

# --- Sidebar Setup ---
with st.sidebar:
    st.header("🔧 Simulation Settings")
    
    with st.expander("ℹ️ About Strategies", expanded=False):
        st.markdown("""
            - **Q-learning & PPO:** Use Reinforcement Learning agents to make pit stop decisions. Requires retrained agents compatible with the current environment.
            - **Head-to-Head:** Compares Q-learning vs. PPO for a single race.
            - **Custom Strategy:** Manually define specific laps for pit stops.
            - **Statistical Comparison:** Run many races per strategy to compare their performance distributions.
        """)
    strategy = st.selectbox(
        "Choose a strategy", 
        ["Q-learning", "PPO", "Head-to-Head", "Custom Strategy"], 
        key="strategy_choice",
        help="Select the decision-making logic for pit stops and tire choices."
    )

    st.subheader("📆 Simulation Mode")
    session_type_selection = st.selectbox(
        "Session Type", 
        ["Race", "Full Weekend", "Head-to-Head", "Statistical Comparison", "Practice", "Quali"], 
        key="session_type_sb",
        help="Choose a single session or a more advanced analysis mode."
    )
    
    # --- Conditional UI for Statistical Comparison ---
    strategies_to_compare = []
    num_runs_per_strategy = 1
    if session_type_selection == "Statistical Comparison":
        st.subheader("📊 Comparison Settings")
        st.info("This mode runs many race simulations per strategy to compare their performance distributions.")
        strategies_to_compare = st.multiselect(
            "Select strategies to compare",
            ["Q-learning", "PPO", "Custom Strategy"], 
            default=["Q-learning", "PPO"],
            key="strategies_to_compare_ms",
            help="Choose two or more strategies for a statistical showdown."
        )
        num_runs_per_strategy = st.number_input(
            "Simulations per strategy",
            min_value=2, max_value=100, value=10,
            key="num_runs_per_strategy_ni",
            help="How many times to run a full race for each selected strategy. Higher numbers give better statistical insight but take longer."
        )

    total_laps = st.slider(
        "Total Laps (for Race)", 20, 80, 58, 
        key="total_laps_slider",
        help="Set the number of laps for the main Race session."
    )

    st.subheader("️🛤️ Track Settings")
    track_options = {
        "Monza": {"pit_time": 28, "tire_wear_rate_config": 1.1, "traffic_penalty_config": 3.0, "base_lap_time": 80.0},
        "Spa": {"pit_time": 32, "tire_wear_rate_config": 1.2, "traffic_penalty_config": 4.0, "base_lap_time": 105.0},
        "Monaco": {"pit_time": 25, "tire_wear_rate_config": 1.4, "traffic_penalty_config": 7.5, "base_lap_time": 71.0},
        "Bahrain": {"pit_time": 30, "tire_wear_rate_config": 2.0, "traffic_penalty_config": 5.0, "base_lap_time": 92.0},
        "Silverstone": {"pit_time": 29, "tire_wear_rate_config": 1.8, "traffic_penalty_config": 4.5, "base_lap_time": 88.0},
        "Suzuka": {"pit_time": 30, "tire_wear_rate_config": 1.7, "traffic_penalty_config": 5.5, "base_lap_time": 91.0},
        "Interlagos": {"pit_time": 26, "tire_wear_rate_config": 1.5, "traffic_penalty_config": 4.0, "base_lap_time": 72.0},
        "COTA": {"pit_time": 28, "tire_wear_rate_config": 1.6, "traffic_penalty_config": 4.2, "base_lap_time": 96.0},
        "Zandvoort": {"pit_time": 27, "tire_wear_rate_config": 1.3, "traffic_penalty_config": 6.5, "base_lap_time": 73.0},
        "Custom": {}
    }
    selected_track_name = st.selectbox("Choose a Track", list(track_options.keys()), key="track_choice", help="Select a predefined track or 'Custom'.")
    
    base_lap_time_cfg = 90.0
    if selected_track_name == "Custom":
        pit_time_config = st.slider("Pit Stop Time (s)", 20, 40, 28, key="custom_pit_time", help="Base time lost for a pit stop.")
        tire_wear_rate_cfg = st.slider("Track Abrasiveness Factor", 0.5, 2.5, 1.5, key="custom_tire_wear", help="Higher = faster tire wear.")
        traffic_penalty_cfg = st.slider("Traffic Penalty Factor", 1.0, 10.0, 5.0, key="custom_traffic", help="Higher = more time lost in traffic.")
        base_lap_time_cfg = st.slider("Base Lap Time (s)", 60, 120, 90, key="custom_base_lap_time", help="Ideal dry lap time for this custom track.")
    else:
        track_props = track_options[selected_track_name]
        pit_time_config = track_props["pit_time"]
        tire_wear_rate_cfg = track_props["tire_wear_rate_config"]
        traffic_penalty_cfg = track_props["traffic_penalty_config"]
        base_lap_time_cfg = track_props.get("base_lap_time", 90.0) 
        st.caption(f"Selected: {selected_track_name} (Pit Time: {pit_time_config}s, Abrasiveness: {tire_wear_rate_cfg}, Traffic: {traffic_penalty_cfg}, Base Lap: {base_lap_time_cfg}s)")

    st.subheader("🌦️ Race Events")
    use_forecast = st.checkbox("🌧️ Use Probabilistic Rain Forecast", key="use_forecast_checkbox", help="If checked, rain may occur within specified lap windows. If unchecked, set fixed rain laps.")
    rain_forecast_ranges = []; rain_laps_fixed = [] 
    if use_forecast:
        st.markdown("###### Forecasted Rain Windows")
        num_forecasts = st.slider("Number of Forecast Periods", 1, 3, 1, key="num_forecasts_slider", help="Define multiple potential rain windows.")
        for i in range(num_forecasts):
            st.markdown(f"**Forecast Window {i+1}**")
            cols_rain = st.columns(3)
            with cols_rain[0]: start_lap_f = st.number_input(f"S{i+1}", 1, total_laps, max(1,min(15,total_laps-5 if total_laps >5 else total_laps)), key=f"fs_{i}", help="Lap rain might start.")
            with cols_rain[1]: end_lap_f = st.number_input(f"E{i+1}", start_lap_f, total_laps, max(start_lap_f,min(start_lap_f+5, total_laps)), key=f"fe_{i}", help="Lap this rain window might end.")
            with cols_rain[2]: prob_f = st.slider(f"C{i+1}%", 0, 100, 50, key=f"fp_{i}", help="Chance of rain in this window.")
            rain_forecast_ranges.append({"start": start_lap_f, "end": end_lap_f, "probability": prob_f / 100.0, "intensity": random.uniform(0.3, 0.8)})
    else:
        rain_laps_fixed = st.multiselect("Apply Rain Mid-Race (Fixed Laps)", list(range(1, total_laps + 1)), key="fixed_rain_laps", help="Manually select laps where rain will occur.")
    safety_car_laps = st.multiselect("Deploy Safety Car on Laps", list(range(1, total_laps + 1)), key="sc_laps", help="Select laps for SC deployment.")
    initial_tire_type_choice = st.selectbox("Initial Tire Type (Race)", ["soft", "medium", "hard"], index=1, key="initial_tire", help="Starting tire for the Race session.")
    
    st.subheader("🏎️ Team & Driver Context")
    selected_team_name = st.selectbox("Choose a team", list(teams.keys()), key="team_choice")
    with st.expander("ℹ️ About Driver Profiles", expanded=False):
        st.markdown("""Driver profiles influence pit strategy and performance:
                    - **Aggressive:** Higher tire wear threshold before pitting. May gain small time bonuses.
                    - **Balanced:** Standard thresholds.
                    - **Conservative:** Lower tire wear threshold (pits earlier). May be slightly more cautious.""")
    selected_driver_profile_name = st.selectbox("Driver Profile", list(driver_profiles.keys()), key="driver_profile", help="Select a driver profile.")
    
    custom_pit_laps_input = []
    if strategy == "Custom Strategy":
        st.subheader("🛠️ Custom Strategy")
        custom_pit_laps_input = st.multiselect("Select Pit Laps", list(range(1, total_laps + 1)), key="custom_pits", help="Define laps for pit stops (defaults to Medium tires if agent is not choosing).")
    
    st.subheader("🎛️ Replay Speed")
    replay_delay = st.slider("Animation Speed (s/lap)", 0.001, 0.5, 0.05, step=0.001, key="replay_delay_slider", format="%.3f", help="Lower is faster.")
# --- End of Sidebar Setup ---

start_button = st.button("▶️ Start Simulation", type="primary", key="main_start_button")

@st.cache_resource
def load_lap_time_predictor_model(path="models/lap_time_predictor.pkl"):
    if os.path.exists(path):
        try: return joblib.load(path)
        except Exception as e: st.error(f"Error loading ML model ({path}): {e}"); return None
    st.info(f"ML lap time predictor model not found at {path}. 'ML Insights' section will be unavailable.")
    return None
ml_lap_predictor_model = load_lap_time_predictor_model()

if use_forecast and rain_forecast_ranges:
    with st.expander("☁️ Forecasted Rain Probabilities", expanded=False):
        for forecast in rain_forecast_ranges:
            st.markdown(f"- **Laps {forecast['start']}–{forecast['end']}**: {forecast['probability']*100:.0f}% chance (Simulated Intensity if occurs: ~{forecast['intensity']:.2f})")

# --- Helper Functions ---
def determine_actual_rain_events(total_laps_sim, forecast_ranges_sim, fixed_rain_laps_sim):
    actual_rain_map = {} 
    if use_forecast and forecast_ranges_sim: 
        for forecast in forecast_ranges_sim:
            if random.random() < forecast["probability"]:
                for lap in range(forecast["start"], forecast["end"] + 1): actual_rain_map[lap] = forecast["intensity"] 
    elif fixed_rain_laps_sim:
        for lap in fixed_rain_laps_sim: actual_rain_map[lap] = random.uniform(0.4, 0.7) 
    return actual_rain_map

def create_simulation_env(tl, base_lt, pt, twr_cfg, tpc_cfg):
    return PitStopEnv(total_laps=tl, base_lap_time_seconds=base_lt, pit_time=pt, 
                      tire_wear_rate_config=twr_cfg, traffic_penalty_config=tpc_cfg)

# --- Core Simulation Function ---
def run_simulation(current_strategy_param, session_type_param, total_laps_param, 
                   pit_time_param, tire_wear_rate_cfg_param, traffic_penalty_cfg_param, 
                   base_lap_time_param, initial_tire_param, 
                   safety_car_laps_param, rain_events_map_param, custom_pit_laps_param, 
                   q_agent_obj_param, ppo_agent_obj_param, driver_profile_obj_param):
    
    # Create a fresh environment for this specific run
    env = create_simulation_env(total_laps_param, base_lap_time_param, pit_time_param, 
                                tire_wear_rate_cfg_param, traffic_penalty_cfg_param)
    
    current_q_agent = q_agent_obj_param
    current_ppo_agent = ppo_agent_obj_param

    if current_strategy_param == "Q-learning":
        if current_q_agent is None:
            st.info(f"Creating new Q-learning agent for '{session_type_param}' session.")
            current_q_agent = QAgent(env)
        elif hasattr(current_q_agent, 'set_env'):
            current_q_agent.set_env(env)
            
    elif current_strategy_param == "PPO" and current_ppo_agent is None:
        st.warning(f"PPO strategy selected for '{session_type_param}', but no PPO model loaded.")

    def ensure_df_columns(df_log_internal):
        expected_cols = ["lap", "lap_time", "tire_type", "tire_wear", "traffic", "safety_car_active", 
                         "rain", "track_temperature", "grip_factor", "vsc_active", "action", 
                         "fuel_weight", "rain_intensity"]
        for col in expected_cols:
            if col not in df_log_internal.columns:
                if col in ["lap_time", "tire_wear", "traffic", "fuel_weight", "track_temperature", "grip_factor", "rain_intensity"]: df_log_internal[col] = pd.NA
                elif col in ["safety_car_active", "rain", "vsc_active", "action"]: df_log_internal[col] = 0
                else: df_log_internal[col] = "unknown" if col == "tire_type" else None
        return df_log_internal

    if session_type_param == "Practice":
        practice_lap_times_list = []; num_stints, laps_per_stint = 3, 3; env_practice_log_list = []
        for i in range(num_stints):
            obs_p, _ = env.reset(options={"initial_tire_type": np.random.choice(["soft", "medium", "hard"])})
            for _ in range(laps_per_stint):
                action_p = 0
                if current_strategy_param == "Q-learning" and current_q_agent: action_p = current_q_agent.choose_action(current_q_agent.discretize(obs_p))
                elif current_strategy_param == "PPO" and current_ppo_agent: pred_p, _ = current_ppo_agent.predict(obs_p, deterministic=True); action_p = int(pred_p.item())
                obs_p, reward_p, done_p, truncated_p, _ = env.step(action_p)
                if not (done_p or truncated_p): practice_lap_times_list.append(-reward_p)
                else: break
            if hasattr(env, 'lap_log'): env_practice_log_list.extend(env.lap_log)
            if env.current_lap >= total_laps_param : break 
        df_log = ensure_df_columns(pd.DataFrame(env_practice_log_list))
        return {"lap_times_data": practice_lap_times_list}, df_log

    elif session_type_param == "Quali":
        best_q_time = float('inf'); all_q_laps = []
        obs_q, _ = env.reset(options={"initial_tire_type": initial_tire_param})
        for _ in range(3): 
            action_q = 0
            if current_strategy_param == "Q-learning" and current_q_agent: action_q = current_q_agent.choose_action(current_q_agent.discretize(obs_q))
            elif current_strategy_param == "PPO" and current_ppo_agent: pred_q, _ = current_ppo_agent.predict(obs_q, deterministic=True); action_q = int(pred_q.item())
            obs_q, r_q, d_q, t_q, _ = env.step(action_q); lap_t_q = -r_q; all_q_laps.append(lap_t_q)
            if action_q == 0 and lap_t_q < best_q_time: best_q_time = lap_t_q
            if d_q or t_q or env.current_lap >= total_laps_param: break
        df_log = ensure_df_columns(pd.DataFrame(env.lap_log if hasattr(env, 'lap_log') else []))
        return {"quali_laps_data": all_q_laps, "best_quali_time": best_q_time}, df_log

    # --- RACE SESSION ---
    obs_race, _ = env.reset(options={"initial_tire_type": initial_tire_param})
    used_compounds_race_set = {initial_tire_param}
    lap_data_race_list = []
    
    for _lap_num_race_iter in range(1, total_laps_param + 1):
        current_lap_for_events = env.current_lap + 1 
        if current_lap_for_events in rain_events_map_param:
            env.apply_rain(intensity=rain_events_map_param[current_lap_for_events])
        elif env.rain_active and current_lap_for_events not in rain_events_map_param: env.clear_rain()
        if current_lap_for_events in safety_car_laps_param: env.apply_safety_car(duration=random.randint(2,3))

        action_race = 0 
        if current_strategy_param == "Q-learning" and current_q_agent:
            state_race = current_q_agent.discretize(obs_race); action_race = current_q_agent.choose_action(state_race)
        elif current_strategy_param == "PPO" and current_ppo_agent:
            pred_r, _ = current_ppo_agent.predict(obs_race, deterministic=True); action_race = int(pred_r.item())
        elif current_strategy_param == "Custom Strategy":
            if current_lap_for_events in custom_pit_laps_param: action_race = 2 
            else: action_race = 0 
        
        if action_race == 0 and isinstance(obs_race, np.ndarray) and len(obs_race) > 1 and obs_race[1] > driver_profile_obj_param["pit_threshold"]:
            action_race = 2
        
        next_obs_race, reward_race, terminated_race, truncated_race, _ = env.step(action_race)
        if action_race > 0: used_compounds_race_set.add(env.current_tire_type)
            
        effective_reward_race = reward_race - driver_profile_obj_param["overtake_bonus"] 
        lap_data_race_list.append((
            env.current_lap, action_race, 
            next_obs_race[1] if len(next_obs_race) > 1 else 0.0, 
            next_obs_race[2] if len(next_obs_race) > 2 else 0.0, 
            effective_reward_race, 
            next_obs_race[3] if len(next_obs_race) > 3 else 0.0 ))
        obs_race = next_obs_race
        if terminated_race or truncated_race: break

    fia_penalty = 0; dry_used_r = used_compounds_race_set & {"soft","medium","hard"}
    race_had_rain = False 
    if hasattr(env, 'lap_log') and env.lap_log:
        df_temp = pd.DataFrame(env.lap_log)
        if 'rain' in df_temp.columns and df_temp['rain'].any(): race_had_rain = True
    if not race_had_rain and len(dry_used_r)<2 and total_laps_param>10: fia_penalty=20
    
    df_log = ensure_df_columns(pd.DataFrame(env.lap_log if hasattr(env, 'lap_log') else []))
    return lap_data_race_list, (env.race_event_messages if hasattr(env,'race_event_messages') else []), fia_penalty, df_log, used_compounds_race_set

# --- Plotly Chart Function Definitions ---
def show_lap_delta_chart_plotly(df_chart_log, title_suffix=""):
    st.markdown(f"### 🟡 Lap Time Delta {title_suffix}") 
    if "lap_time" in df_chart_log.columns and not df_chart_log.empty and df_chart_log["lap_time"].notna().any():
        df_copy = df_chart_log.copy().dropna(subset=['lap_time']) 
        if df_copy.empty: st.info("Not enough valid lap time data for Delta chart."); return
        base_lap_time = df_copy["lap_time"].iloc[0]
        df_copy["lap_delta"] = df_copy["lap_time"] - base_lap_time
        fig = px.line(df_copy, x="lap", y="lap_delta", title="Lap Delta from First Valid Lap", labels={"lap": "Lap", "lap_delta": "Delta (s)"}, markers=True)
        fig.add_hline(y=0, line_dash="dash", line_color="gray"); st.plotly_chart(fig, use_container_width=True)
    else: st.info("Lap Time data not available for Delta chart.")

def show_tire_usage_chart_plotly(df_chart_log, title_suffix=""):
    st.markdown(f"### 🔵 Tire Compound Usage {title_suffix}")
    if "tire_type" in df_chart_log.columns and df_chart_log["tire_type"].notna().any():
        compound_counts = df_chart_log["tire_type"].value_counts().reset_index(); compound_counts.columns = ['tire_type', 'count']
        fig = px.bar(compound_counts, x='tire_type', y='count', title="Laps per Tire Compound", labels={'tire_type': 'Tire Type', 'count': 'Laps Used'}, color='tire_type', color_discrete_map={"soft": "#FF6347", "medium": "#FFD700", "hard": "#D3D3D3", "intermediate": "#32CD32", "wet": "#1E90FF"})
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Tire Type data not available for Usage chart.")

def show_track_temperature_chart_plotly(df_chart_log, title_suffix=""):
    st.markdown(f"### 🔥 Track Temperature {title_suffix}")
    if "track_temperature" in df_chart_log.columns and df_chart_log["track_temperature"].notna().any():
        fig = px.line(df_chart_log, x="lap", y="track_temperature", title="Track Temperature (°C)", labels={"lap": "Lap", "track_temperature": "Temp (°C)"}, color_discrete_sequence=["orange"])
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Track Temperature data not available.")

def show_grip_factor_chart_plotly(df_chart_log, title_suffix=""):
    st.markdown(f"### 🟢 Track Grip Factor {title_suffix}")
    if "grip_factor" in df_chart_log.columns and df_chart_log["grip_factor"].notna().any():
        fig = px.line(df_chart_log, x="lap", y="grip_factor", title="Track Grip Factor", labels={"lap": "Lap", "grip_factor": "Grip"}, color_discrete_sequence=["green"])
        if not df_chart_log["grip_factor"].empty and df_chart_log["grip_factor"].notna().any():
            min_grip = df_chart_log["grip_factor"].min(); max_grip = df_chart_log["grip_factor"].max()
            if pd.notna(min_grip) and pd.notna(max_grip) and min_grip != max_grip : fig.update_yaxes(range=[min_grip * 0.95, max_grip * 1.05])
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Grip Factor data not available.")

def show_ml_predictions_plotly(df_with_predictions, title_suffix=""):
    st.markdown(f"### ⏱️ ML: Actual vs Predicted Lap Times {title_suffix}")
    if all(col in df_with_predictions.columns for col in ["lap", "lap_time", "predicted_lap_time"]) and df_with_predictions["lap_time"].notna().any() and df_with_predictions["predicted_lap_time"].notna().any():
        plot_df = df_with_predictions[["lap", "lap_time", "predicted_lap_time"]].melt(id_vars=['lap'], var_name='Lap Time Type', value_name='Time (s)')
        fig = px.line(plot_df, x="lap", y="Time (s)", color='Lap Time Type', title="ML: Actual vs. Predicted Lap Times", markers=True, line_dash_map={"lap_time":"solid", "predicted_lap_time":"dash"})
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Sufficient actual or predicted lap time data missing for ML comparison chart.")

def animate_lap_chart_plotly(lap_data_anim, strategy_name_anim, team_color_hex, total_laps_anim, selected_team_anim, selected_profile_anim, replay_delay_anim, pit_time_race_anim, fia_penalty_anim):
    st.markdown(f"### {selected_team_anim} | {selected_profile_anim} Driver: {strategy_name_anim} Lap Metrics Animation")
    with st.expander("💡 Understanding the Lap Metrics Animation", expanded=False):
        st.markdown("""This chart visualizes key metrics for each lap:
                    - **Tire Wear (%):** Resets after a pit stop (action > 0).
                    - **Smoothed Tire Wear:** Rolling average for trend spotting.
                    - **Traffic Intensity (%):** Simulated effect of other cars.
                    - **Fuel Weight (kg):** Decreases each lap.
                    - **Red Dotted Lines:** Indicate pit stop laps (where action > 0).""")
    fig = go.Figure(); trace_names = ['Tire Wear (%)', 'Smoothed Tire Wear', 'Traffic Intensity (%)', 'Fuel Weight (kg)']; trace_colors = [team_color_hex, 'blue', 'grey', 'green']; trace_dash_styles = ['solid', 'dash', 'dash', 'dot']
    for i, name in enumerate(trace_names): fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=name, line=dict(color=trace_colors[i], dash=trace_dash_styles[i])))
    fig.update_layout(xaxis_title="Lap", yaxis_title="Value", xaxis=dict(range=[0, total_laps_anim]), yaxis=dict(range=[0, 110]), legend_title_text='Metrics', title_x=0.5, height=500)
    chart_placeholder = st.empty()
    x_laps, tire_wear_vals, traffic_vals, fuel_vals = [], [], [], []; current_shapes = []
    for lap, action, tire_wear, traffic_pcnt, reward, fuel_kg in lap_data_anim: # Action is now 0-5
        x_laps.append(lap); tire_wear_vals.append(tire_wear); traffic_vals.append(traffic_pcnt * 100 if traffic_pcnt is not None and traffic_pcnt <= 1.0 else (traffic_pcnt if traffic_pcnt is not None else 0)); fuel_vals.append(fuel_kg)
        smoothed_wear_vals = pd.Series(tire_wear_vals).rolling(window=min(3, len(tire_wear_vals)), min_periods=1).mean().tolist()
        if action > 0: current_shapes.append(dict(type="line", x0=lap, y0=0, x1=lap, y1=110, line=dict(color="#FF0000", width=1.5, dash="dashdot"), name=f"Pit Lap {lap}"))
        fig_update = go.Figure(); fig_update.add_trace(go.Scatter(x=x_laps, y=tire_wear_vals, mode='lines', name=trace_names[0], line=dict(color=trace_colors[0], dash=trace_dash_styles[0]))); fig_update.add_trace(go.Scatter(x=x_laps, y=smoothed_wear_vals, mode='lines', name=trace_names[1], line=dict(color=trace_colors[1], dash=trace_dash_styles[1]))); fig_update.add_trace(go.Scatter(x=x_laps, y=traffic_vals, mode='lines', name=trace_names[2], line=dict(color=trace_colors[2], dash=trace_dash_styles[2]))); fig_update.add_trace(go.Scatter(x=x_laps, y=fuel_vals, mode='lines', name=trace_names[3], line=dict(color=trace_colors[3], dash=trace_dash_styles[3])));
        fig_update.update_layout(xaxis_title="Lap", yaxis_title="Value", xaxis=dict(range=[0, total_laps_anim]), yaxis=dict(range=[0, 110]), legend_title_text='Metrics', title_x=0.5, height=500, shapes=current_shapes)
        chart_placeholder.plotly_chart(fig_update, use_container_width=True); time.sleep(replay_delay_anim)
    
    st.markdown(f"🚗 **Pit Stops occurred at Laps:** { [l for l,a,*_ in lap_data_anim if a > 0] }")
    avg_tire = float(np.mean([t for _, _, t, _, _, _ in lap_data_anim])) if lap_data_anim else 0
    avg_traffic_raw = float(np.mean([x for _, _, _, x, _, _ in lap_data_anim])) if lap_data_anim else 0
    avg_traffic = avg_traffic_raw * 100 if avg_traffic_raw is not None and avg_traffic_raw <=1.0 else avg_traffic_raw
    avg_fuel = float(np.mean([f for *_, f in lap_data_anim])) if lap_data_anim else 0
    total_reward_val = float(sum([r for *_, r, _ in lap_data_anim])) - fia_penalty_anim if lap_data_anim else -fia_penalty_anim
    num_pit_stops_val = len([l for l,a,*_ in lap_data_anim if a > 0])
    pit_eff = 0
    if total_laps_anim > 0 and pit_time_race_anim > 0:
        race_dur_est = total_laps_anim * base_lap_time_cfg 
        pit_loss_est = num_pit_stops_val * pit_time_race_anim
        if race_dur_est > 0: pit_eff = max(0, 100 - (pit_loss_est/race_dur_est)*100)

    st.subheader("📊 Post-Race Strategy Review") 
    with st.expander("What do these metrics mean?", expanded=True):
        st.markdown("""
            - **Total Reward (After FIA Penalty):** Sum of per-lap rewards (lap times are negative rewards) plus bonuses, minus penalties. Higher is generally better.
            - **Avg Tire Wear:** Average wear percentage across all tire stints.
            - **Avg Traffic Level:** Average traffic intensity faced by the driver.
            - **Avg Fuel Weight:** Average fuel load during the race.
            - **Pit Efficiency Rating (Estimate):** A conceptual measure of time lost in pits versus an ideal. Higher is better.
            - **FIA Penalty:** Penalties applied (e.g., for not using required tire compounds in a dry race).
        """)
    st.markdown(f"- **Total Reward (After FIA Penalty):** {total_reward_val:.2f}")
    st.markdown(f"- **Avg Tire Wear:** {avg_tire:.2f}%")
    st.markdown(f"- **Avg Traffic Level:** {avg_traffic:.2f}%")
    st.markdown(f"- **Avg Fuel Weight:** {avg_fuel:.2f}kg")
    st.markdown(f"- **Pit Efficiency Rating (Estimate):** {pit_eff:.1f}%")
    if fia_penalty_anim > 0: st.markdown(f"- 🚫 **FIA Penalty Applied:** -{fia_penalty_anim} (units/seconds)")


def show_decision_timeline_plotly(lap_data_timeline, df_log_for_timeline, total_laps_timeline):
    st.subheader("🧠 Strategic Event Timeline")
    if df_log_for_timeline is None or df_log_for_timeline.empty: st.info("No detailed log data for decision timeline."); return
    with st.expander("💡 Understanding the Timeline Icons", expanded=False):
        st.markdown("""This timeline highlights key events on specific laps:
                    - 🅿️: Pit Stop made. - 🌧️: Rain active. - ⚠️: Safety Car active. - 🚦: Virtual Safety Car (VSC) active.""")
    pit_laps_tl = [lap for lap, action, *_ in lap_data_timeline if action > 0] 
    rain_laps_tl = df_log_for_timeline[df_log_for_timeline["rain"] == True]["lap"].tolist() if "rain" in df_log_for_timeline.columns else []
    sc_laps_tl = df_log_for_timeline[df_log_for_timeline["safety_car_active"] == True]["lap"].tolist() if "safety_car_active" in df_log_for_timeline.columns else []
    vsc_laps_tl = df_log_for_timeline[df_log_for_timeline["vsc_active"] == True]["lap"].tolist() if "vsc_active" in df_log_for_timeline.columns else []
    all_event_laps_set = set(pit_laps_tl + rain_laps_tl + sc_laps_tl + vsc_laps_tl)
    if not all_event_laps_set: st.info("No specific events (Pits, Rain, SC, VSC) to show on timeline."); return
    all_event_laps = sorted(list(all_event_laps_set)); timeline_df_data = []
    for lap_event in all_event_laps:
        icons = ""; 
        if lap_event in pit_laps_tl: icons += "🅿️"
        if lap_event in rain_laps_tl: icons += "🌧️"
        if lap_event in sc_laps_tl: icons += "⚠️"
        if lap_event in vsc_laps_tl: icons += "🚦"
        timeline_df_data.append({"lap": lap_event, "y": 0.5, "icons": icons.strip()})
    if not timeline_df_data: st.info("No events to plot on timeline."); return
    timeline_df = pd.DataFrame(timeline_df_data)
    fig = px.scatter(timeline_df, x="lap", y="y", text="icons", title="Strategic Event Timeline", labels={"lap": "Lap Number", "y":""})
    fig.update_traces(textfont_size=16); fig.update_layout(yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5,1.5]), xaxis=dict(range=[0, total_laps_timeline + 1], autorange=False), height=150, title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

def generate_strategy_pdf(strategy_pdf, total_laps_pdf, team_pdf, profile_pdf, pit_markers_pdf, compounds_pdf, summary_pdf):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "F1 Strategy Report", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C"); pdf.ln(5) # Use Helvetica & new_x/new_y
    page_width = pdf.w - pdf.l_margin - pdf.r_margin 
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(page_width, 7, f"Team: {team_pdf} | Driver Profile: {profile_pdf} | Strategy: {strategy_pdf} | Total Laps: {total_laps_pdf}"); pdf.ln(2)
    pdf.multi_cell(page_width, 7, f"Pit Stops occurred at Laps: {', '.join(map(str, sorted(pit_markers_pdf))) if pit_markers_pdf else 'None'}"); pdf.ln(2)
    pdf.multi_cell(page_width, 7, f"Tire Compounds Used: {', '.join(sorted(list(compounds_pdf)))}"); pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13); pdf.cell(page_width, 10, "Performance Summary:", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.set_font("Helvetica", "", 11)
    for key, val in summary_pdf.items(): pdf.multi_cell(page_width, 7, f"  {key}: {str(val)}")
    return bytes(pdf.output())

def show_race_replay(total_laps_rep, team_color_rep, team_icon_rep, replay_delay_rep):
    st.subheader("🎥 Mini Race Replay"); fig, ax = plt.subplots(figsize=(3.5, 3.5)); track_radius = 1.0; theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(track_radius * np.cos(theta), track_radius * np.sin(theta), "--", lw=2, color="grey"); car_dot, = ax.plot([], [], 'o', color=team_color_rep, markersize=10, label=team_icon_rep)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis('off'); ax.legend(loc='center', fontsize='small'); replay_placeholder = st.empty()
    for lap_r in range(1, int(total_laps_rep) + 1): angle = (2 * np.pi * lap_r / total_laps_rep) - (np.pi/2); car_dot.set_data([track_radius * np.cos(angle)], [track_radius * np.sin(angle)]); replay_placeholder.pyplot(fig); time.sleep(replay_delay_rep)
    plt.close(fig)

# --- ML Insights Block (Reusable Function) ---
def display_ml_insights(df_log_for_ml, model_to_use, title_suffix=""):
    if model_to_use and df_log_for_ml is not None and not df_log_for_ml.empty:
        with st.expander(f"💡 About ML Lap Time Predictions {title_suffix}", expanded=False):
            st.markdown("""This section uses a pre-trained ML model (`lap_time_predictor.pkl`)
                        to predict lap times based on simulated conditions for each lap of this race.
                        It compares these predictions to the actual simulated lap times. This helps assess
                        the ML model's accuracy and understand which factors it deems important.""")
        st.markdown(f"### 🔮 ML Insights: Lap Time Prediction {title_suffix}")
        try:
            features_ml = df_log_for_ml.copy(); bool_cols = ['rain', 'safety_car_active', 'vsc_active']
            for col in bool_cols:
                if col in features_ml.columns:
                    if features_ml[col].dtype == 'object': features_ml[col] = features_ml[col].astype(str).str.lower().map({'true':1,'false':0,'1':1,'0':0,'1.0':1,'0.0':0}).fillna(0).astype(int)
                    else: features_ml[col] = features_ml[col].astype(bool).astype(int)
                else: features_ml[col] = 0
            if "tire_type" in features_ml.columns:
                features_ml = pd.concat([features_ml, pd.get_dummies(features_ml["tire_type"], prefix="tire_type", dummy_na=False)], axis=1)
            model_trained_features = ['lap','tire_wear','traffic','fuel_weight','track_temperature','grip_factor','rain','safety_car_active','vsc_active','tire_type_soft','tire_type_medium','tire_type_hard','tire_type_intermediate','tire_type_wet']
            X_ml_dict = {}
            for col_name in model_trained_features:
                X_ml_dict[col_name] = features_ml[col_name] if col_name in features_ml.columns else (pd.Series([0]*len(features_ml),index=features_ml.index) if not features_ml.empty else pd.Series([0]))
            
            if features_ml.empty and not X_ml_dict : X_ml = pd.DataFrame(columns=model_trained_features)
            else: X_ml = pd.DataFrame(X_ml_dict)

            if not model_trained_features: st.warning("Model trained features list is empty."); return
            X_ml = X_ml[model_trained_features] 
            
            if X_ml.isnull().values.any():
                for col_n in X_ml.columns:
                    if X_ml[col_n].isnull().any():
                        if pd.api.types.is_numeric_dtype(X_ml[col_n]): X_ml[col_n].fillna(X_ml[col_n].median(),inplace=True)
                        else: X_ml[col_n].fillna(0,inplace=True)
            if not X_ml.empty:
                predicted_laps = model_to_use.predict(X_ml); features_ml["predicted_lap_time"] = predicted_laps
                show_ml_predictions_plotly(features_ml, title_suffix)
            else: st.info("Not enough data for ML predictions.")
        except Exception as e: st.error(f"Error in ML Insights: {e}"); st.exception(e)

# --- Main Application Logic ---
if start_button:
    current_driver_profile = driver_profiles[selected_driver_profile_name]
    actual_rain_event_map = determine_actual_rain_events(total_laps, rain_forecast_ranges, rain_laps_fixed)
    
    q_agent_loaded_instance = None 
    if strategy == "Q-learning" or session_type_selection == "Head-to-Head":
        q_path = f"saved_agents/{selected_team_name}_{selected_driver_profile_name}_q.pkl"
        if os.path.exists(q_path):
            try:
                with open(q_path, "rb") as f: q_agent_loaded_instance = pickle.load(f)
                st.info(f"Loaded Q-agent: {q_path}")
            except Exception as e: st.error(f"Error loading Q-agent {q_path}: {e}")
        else: st.info(f"No saved Q-agent at {q_path}. New Q-agent will be used if Q-learning strategy is selected.")

    ppo_agent_loaded_instance = None
    if strategy == "PPO" or session_type_selection == "Head-to-Head":
        ppo_path = "models/ppo_pit_stop.zip"
        if os.path.exists(ppo_path):
            try:
                ppo_agent_loaded_instance = PPO.load(ppo_path) 
                st.info(f"Loaded PPO model: {ppo_path}")
            except Exception as e: st.error(f"Error loading PPO model {ppo_path}: {e}")
        else: st.warning(f"PPO model not found at {ppo_path}. PPO strategy may be unavailable.")

    base_sim_params = [pit_time_config, tire_wear_rate_cfg, traffic_penalty_cfg, base_lap_time_cfg]
    event_params = [safety_car_laps, actual_rain_event_map, custom_pit_laps_input]
    agent_params = [q_agent_loaded_instance, ppo_agent_loaded_instance, current_driver_profile]

    # --- FULL WEEKEND LOGIC ---
    if session_type_selection == "Full Weekend":
        # ... (Full Weekend logic as defined before) ...
        st.header(f"🏁 Full Weekend: {selected_track_name} - {selected_team_name} ({strategy})")
        if st.session_state.weekend_phase == "Practice":
            st.subheader("🔧 Practice Session");
            with st.spinner("Simulating Practice..."):
                results_display, df_log_display = run_simulation(strategy, "Practice", 15, *base_sim_params, "soft", [], {}, [], *agent_params)
            if results_display and "lap_times_data" in results_display: st.write("Practice Lap Times Focus:", [round(t,3) for t in results_display["lap_times_data"]])
            if not df_log_display.empty: show_tire_usage_chart_plotly(df_log_display, "(Practice)")
            st.session_state.weekend_phase = "Quali"; st.success("✅ Practice Complete."); st.rerun()
        elif st.session_state.weekend_phase == "Quali":
            st.subheader("⏱️ Qualifying Session"); quali_tire = "soft" if not any(actual_rain_event_map.values()) else "intermediate"
            with st.spinner("Simulating Qualifying..."):
                results_display, df_log_display = run_simulation(strategy, "Quali", 5, *base_sim_params, quali_tire, [], {}, [], *agent_params)
            if results_display:
                st.write("Quali Laps Focus:", [round(t,3) for t in results_display.get("quali_laps_data", [])])
                st.write("Best Quali Time:", f"{results_display.get('best_quali_time', 'N/A'):.3f}s" if isinstance(results_display.get('best_quali_time'), float) else "N/A")
            if not df_log_display.empty: show_lap_delta_chart_plotly(df_log_display, "(Quali)")
            st.session_state.weekend_phase = "Race"; st.success("✅ Qualifying Complete."); st.rerun()
        elif st.session_state.weekend_phase == "Race":
            st.subheader("🏆 Race Simulation")
            lap_data_display, race_msgs_display, penalty_display, df_log_display, used_compounds_display = None, [], 0, pd.DataFrame(), set()
            with st.spinner(f"Simulating Race ({total_laps} Laps)..."):
                try:
                    lap_data_display, race_msgs_display, penalty_display, df_log_display, used_compounds_display = run_simulation(
                        strategy, "Race", total_laps, *base_sim_params, initial_tire_type_choice, *event_params, *agent_params)
                except Exception as e: st.error(f"A critical error occurred during the race simulation: {e}"); st.exception(e)
            if not lap_data_display: st.error("Race simulation failed.")
            else:
                st.success("🏁 Race Simulation Finished!")
                animate_lap_chart_plotly(lap_data_display, strategy, teams[selected_team_name]["color"], total_laps, selected_team_name, selected_driver_profile_name, replay_delay, pit_time_config, penalty_display)
                show_decision_timeline_plotly(lap_data_display, df_log_display, total_laps)
                col_a, col_b = st.columns(2);
                with col_a: show_lap_delta_chart_plotly(df_log_display, "(Race)")
                with col_b: show_tire_usage_chart_plotly(df_log_display, "(Race)")
                col_c, col_d = st.columns(2);
                with col_c: show_track_temperature_chart_plotly(df_log_display, "(Race)")
                with col_d: show_grip_factor_chart_plotly(df_log_display, "(Race)")
                display_ml_insights(df_log_display, ml_lap_predictor_model, "(Race)")
                final_total_reward = sum(r for *_, r, _ in lap_data_display) - penalty_display
                # ... (Summary calc and PDF download as before) ...
                summary_data_pdf = {"Total Reward": f"{final_total_reward:.2f}", "FIA Penalty Applied": f"{penalty_display} (units)"} # Simplified for brevity
                pdf_bytes = generate_strategy_pdf(strategy, total_laps, selected_team_name, selected_driver_profile_name, [l for l,a,*_ in lap_data_display if a > 0], used_compounds_display, summary_data_pdf)
                st.download_button("📄 Download Race Report (PDF)", pdf_bytes, f"{selected_team_name}_{strategy}_FW_RaceReport.pdf", "application/pdf", key="pdf_fw_race")
            st.session_state.weekend_phase = "Practice"; st.info("Full Weekend Simulation Complete.")

    # --- HEAD-TO-HEAD LOGIC ---
    elif session_type_selection == "Head-to-Head":
        # ... (Head-to-Head logic as defined before) ...
        st.header(f"⚔️ Head-to-Head: Q-learning vs PPO at {selected_track_name}")
        col1, col2 = st.columns(2); summary_h2h_list = []
        h2h_race_params = [total_laps, *base_sim_params, initial_tire_type_choice, safety_car_laps, actual_rain_event_map, []] 
        with col1:
            st.subheader("Q-learning Agent")
            with st.spinner("Simulating Q-learning..."):
                q_lap_data, _, q_penalty, q_df_log, q_compounds_h2h = run_simulation("Q-learning", "Race", *h2h_race_params, q_agent_loaded_instance, None, current_driver_profile)
            if q_lap_data: 
                animate_lap_chart_plotly(q_lap_data, "Q-learning", teams[selected_team_name]["color"], total_laps, selected_team_name, selected_driver_profile_name, replay_delay, pit_time_config, q_penalty)
                q_total_r = sum(r for *_,r,_ in q_lap_data) - q_penalty; summary_h2h_list.append({"Strategy": "Q-learning", "Total Reward": f"{q_total_r:.2f}", "Pits": len([l for l,a,*_ in q_lap_data if a>0]), "Compounds": ", ".join(sorted(list(q_compounds_h2h)))})
                with st.expander("Q-Learning Race Charts (H2H)"): show_lap_delta_chart_plotly(q_df_log, "(Q-L H2H)")
            else: st.warning("Q-learning H2H run failed.")
        with col2:
            st.subheader("PPO Agent")
            with st.spinner("Simulating PPO..."):
                ppo_lap_data, _, ppo_penalty, ppo_df_log, ppo_compounds_h2h = run_simulation("PPO", "Race", *h2h_race_params, None, ppo_agent_loaded_instance, current_driver_profile)
            if ppo_lap_data: 
                animate_lap_chart_plotly(ppo_lap_data, "PPO", teams[selected_team_name]["color"], total_laps, selected_team_name, selected_driver_profile_name, replay_delay, pit_time_config, ppo_penalty)
                ppo_total_r = sum(r for *_,r,_ in ppo_lap_data) - ppo_penalty; summary_h2h_list.append({"Strategy": "PPO", "Total Reward": f"{ppo_total_r:.2f}", "Pits": len([l for l,a,*_ in ppo_lap_data if a>0]), "Compounds": ", ".join(sorted(list(ppo_compounds_h2h))) })
                with st.expander("PPO Race Charts (H2H)"): show_lap_delta_chart_plotly(ppo_df_log, "(PPO H2H)")
            else: st.warning("PPO H2H run failed.")
        if summary_h2h_list: st.subheader("📋 Head-to-Head Summary"); st.dataframe(pd.DataFrame(summary_h2h_list).set_index("Strategy"), use_container_width=True)


    # --- STATISTICAL COMPARISON LOGIC ---
    elif session_type_selection == "Statistical Comparison":
        if not strategies_to_compare: st.error("Please select at least one strategy to compare in the sidebar.")
        else:
            st.header(f"📊 Statistical Comparison: {', '.join(strategies_to_compare)}")
            all_results = []
            total_sims_to_run = len(strategies_to_compare) * num_runs_per_strategy
            progress_bar = st.progress(0, text=f"Starting batch simulation for {total_sims_to_run} races...")
            sim_counter = 0
            for strategy_to_test in strategies_to_compare:
                st.subheader(f"Simulating '{strategy_to_test}' strategy...")
                for i in range(num_runs_per_strategy):
                    sim_counter += 1
                    progress_bar.progress(sim_counter / total_sims_to_run, text=f"Running '{strategy_to_test}' simulation {i + 1}/{num_runs_per_strategy}...")
                    
                    # Each run needs a different set of random events
                    current_run_rain_map = determine_actual_rain_events(total_laps, rain_forecast_ranges, rain_laps_fixed)
                    current_run_event_params = [safety_car_laps, current_run_rain_map, custom_pit_laps_input]

                    lap_data, _, penalty, df_log, _ = run_simulation(
                        strategy_to_test, "Race", total_laps, *base_sim_params,
                        initial_tire_type_choice, *current_run_event_params, *agent_params)
                    
                    if lap_data and not df_log.empty:
                        all_results.append({
                            "Strategy": strategy_to_test, "Run": i + 1,
                            "Total Race Time (s)": df_log["lap_time"].sum(),
                            "Total Reward": sum(r for *_, r, _ in lap_data) - penalty,
                            "Number of Pits": len([lap for lap, action, *_ in lap_data if action > 0])})
            progress_bar.progress(1.0, text="Batch simulation complete!")

            if not all_results: st.error("No simulation runs completed successfully.")
            else:
                results_df = pd.DataFrame(all_results)
                st.subheader("📈 Summary Statistics")
                summary_stats = results_df.groupby("Strategy").agg({
                    "Total Race Time (s)": ["mean", "median", "std", "min", "max"],
                    "Number of Pits": ["mean", "std", "min", "max"]}).round(2)
                st.dataframe(summary_stats, use_container_width=True)

                st.subheader("⏱️ Distribution of Total Race Times")
                with st.expander("💡 How to Read a Box Plot", expanded=False):
                    st.markdown("""
                        - The **line inside the box** is the median (50th percentile).
                        - The **box** represents the interquartile range (IQR), from the 25th to the 75th percentile. A smaller box means more consistent results.
                        - The **whiskers** extend to show the rest of the distribution, typically 1.5x the IQR.
                        - **Dots** outside the whiskers are outliers.""")
                fig_box = px.box(results_df, x="Strategy", y="Total Race Time (s)", color="Strategy", points="all", title="Comparison of Total Race Time Distributions")
                st.plotly_chart(fig_box, use_container_width=True)

                st.subheader("🅿️ Distribution of Pit Stops")
                fig_hist = px.histogram(results_df, x="Number of Pits", color="Strategy", barmode="overlay", marginal="rug", title="Comparison of Pit Stop Counts")
                st.plotly_chart(fig_hist, use_container_width=True)


    # --- SINGLE SESSION LOGIC (Practice, Quali, or Race) ---
    else: 
        st.header(f"🚦 Single Session: {session_type_selection} at {selected_track_name} - {selected_team_name} ({strategy})")
        df_log_to_display = pd.DataFrame(); results_data_single = None; 
        lap_data_s, race_msgs_s, penalty_s, used_compounds_s = None, [], 0, set()

        with st.spinner(f"Simulating {session_type_selection}..."):
            laps_for_this_session = total_laps ; initial_tire_this_session = initial_tire_type_choice; current_event_params = event_params
            if session_type_selection == "Practice": laps_for_this_session = 15; initial_tire_this_session = "soft"; current_event_params = [[], {}, []]
            elif session_type_selection == "Quali": laps_for_this_session = 5; initial_tire_this_session = "soft" if not any(actual_rain_event_map.values()) else "intermediate"; current_event_params = [[], {}, []] 
            
            try:
                sim_output = run_simulation(strategy, session_type_selection, laps_for_this_session, 
                                            *base_sim_params, initial_tire_this_session, *current_event_params, *agent_params)
                if session_type_selection in ["Practice", "Quali"]: results_data_single, df_log_to_display = sim_output
                elif session_type_selection == "Race": lap_data_s, race_msgs_s, penalty_s, df_log_to_display, used_compounds_s = sim_output; results_data_single = lap_data_s
            except Exception as e: st.error(f"Error during simulation: {e}"); st.exception(e)

        if session_type_selection == "Practice":
            if results_data_single and "lap_times_data" in results_data_single: st.write("Practice Lap Times:", [round(t,3) for t in results_data_single["lap_times_data"]])
            if not df_log_to_display.empty: show_tire_usage_chart_plotly(df_log_to_display, "(Practice)")
        elif session_type_selection == "Quali":
            if results_data_single:
                st.write("Qualifying Laps:", [round(t,3) for t in results_data_single.get("quali_laps_data", [])])
                st.write("Best Quali Time:", f"{results_data_single.get('best_quali_time', 'N/A'):.3f}s" if isinstance(results_data_single.get('best_quali_time'), float) else "N/A")
            if not df_log_to_display.empty: show_lap_delta_chart_plotly(df_log_to_display, "(Quali)")
        elif session_type_selection == "Race":
            if not results_data_single: st.error("Race simulation failed.") 
            else:
                st.success(f"🏁 {strategy} Race Simulation Finished!")
                animate_lap_chart_plotly(results_data_single, strategy, teams[selected_team_name]["color"], total_laps, selected_team_name, selected_driver_profile_name, replay_delay, pit_time_config, penalty_s)
                show_decision_timeline_plotly(results_data_single, df_log_to_display, total_laps)
                
                col1, col2 = st.columns(2); 
                with col1: show_lap_delta_chart_plotly(df_log_to_display, "(Race)")
                with col2: show_tire_usage_chart_plotly(df_log_to_display, "(Race)")
                col3, col4 = st.columns(2); 
                with col3: show_track_temperature_chart_plotly(df_log_to_display, "(Race)")
                with col4: show_grip_factor_chart_plotly(df_log_to_display, "(Race)")

                display_ml_insights(df_log_to_display, ml_lap_predictor_model, "(Race)")
                
                final_total_reward_s = sum(r for *_, r, _ in results_data_single) - penalty_s
                avg_tire_pdf_s = float(np.mean([t for _, _, t, _, _, _ in results_data_single])) if results_data_single else 0; avg_traffic_pdf_raw_s = float(np.mean([x for _, _, _, x, _, _ in results_data_single])) if results_data_single else 0; avg_traffic_pdf_s = avg_traffic_pdf_raw_s *100 if avg_traffic_pdf_raw_s is not None and avg_traffic_pdf_raw_s <=1.0 else avg_traffic_pdf_raw_s; avg_fuel_pdf_s = float(np.mean([f for *_, f in results_data_single])) if results_data_single else 0; num_pit_stops_pdf_s = len([l for l,a,*_ in results_data_single if a > 0]); pit_efficiency_pdf_s = 0
                if total_laps > 0 and pit_time_config > 0: race_dur_est_pdf_s = total_laps * base_lap_time_cfg ; pit_loss_est_pdf_s = num_pit_stops_pdf_s * pit_time_config;
                if 'race_dur_est_pdf_s' in locals() and race_dur_est_pdf_s > 0 : pit_efficiency_pdf_s = max(0, 100 - (pit_loss_est_pdf_s/race_dur_est_pdf_s)*100)
                summary_data_pdf_s = {"Total Reward": f"{final_total_reward_s:.2f}", "Avg Tire Wear": f"{avg_tire_pdf_s:.2f}%", "Avg Traffic Level": f"{avg_traffic_pdf_s:.2f}%", "Avg Fuel Weight": f"{avg_fuel_pdf_s:.2f}kg", "Pit Efficiency (Estimate)": f"{pit_efficiency_pdf_s:.1f}%", "FIA Penalty Applied": f"{penalty_s} (units)"}
                pdf_bytes_s = generate_strategy_pdf(strategy, total_laps, selected_team_name, selected_driver_profile_name, [l for l,a,*_ in results_data_single if a > 0], used_compounds_s, summary_data_pdf_s)
                st.download_button(
                    "📄 Download Single Race Report (PDF)", 
                    data=pdf_bytes_s,
                    file_name=f"{selected_team_name}_{strategy}_SingleRaceReport.pdf", 
                    mime="application/pdf", 
                    key="pdf_single_race"
                )
