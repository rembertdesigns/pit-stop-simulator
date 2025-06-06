# env/gym_race_env.py (Stable version with 2D Action Space and formulaic lap times)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import csv
import warnings 

class PitStopEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, total_laps=58, base_lap_time_seconds=90.0, pit_time=30, 
                 tire_wear_rate_config=None, traffic_penalty_config=5.0):
        super(PitStopEnv, self).__init__()

        self.total_laps = total_laps
        self.base_track_lap_time = float(base_lap_time_seconds) 
        self.base_pit_time = pit_time 
        self.current_pit_time = pit_time

        self.tire_compound_properties = {
            "soft":       {"wear_multiplier": 1.5, "base_lap_time_offset": 0.0, "grip_factor_bonus": 0.02, "is_rain_tire": False},
            "medium":     {"wear_multiplier": 1.0, "base_lap_time_offset": 0.6, "grip_factor_bonus": 0.0, "is_rain_tire": False},
            "hard":       {"wear_multiplier": 0.7, "base_lap_time_offset": 1.3, "grip_factor_bonus": -0.02, "is_rain_tire": False},
            "intermediate":{"wear_multiplier": 1.2, "base_lap_time_offset": 2.5, "grip_factor_bonus": 0.0, "is_rain_tire": True},
            "wet":        {"wear_multiplier": 0.9, "base_lap_time_offset": 5.0, "grip_factor_bonus": 0.0, "is_rain_tire": True}
        }
        self.track_abrasiveness = tire_wear_rate_config if tire_wear_rate_config is not None else 1.0 
        self.base_traffic_penalty_factor = traffic_penalty_config

        # --- Reverted to 2D Action Space ---
        self.action_space = spaces.Discrete(2)  # 0 = Stay out, 1 = Pit
        
        self.observation_space = spaces.Box(
            low=np.array([0,   0, 0.0,   0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.total_laps, 100, 1.0, 110, 1, 1, 1], dtype=np.float32),
        )
        
        # Initialize properties that will be set by _update_tire_specific_parameters
        self.current_tire_wear_multiplier = 1.0
        self.is_current_tire_rain_tire = False
        self.current_base_lap_time_offset = 0.0
        self.current_tire_grip_bonus = 0.0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 0 
        self.tire_wear = 0.0
        self.traffic = random.uniform(0.05, 0.3) 
        self.fuel_weight = 105.0 
        self.track_temperature = random.uniform(20.0, 30.0) 
        self.base_grip_factor = random.uniform(0.85, 0.95) 
        self.current_grip_factor = self.base_grip_factor
        self.done = False
        self.total_simulated_time = 0.0
        self.current_tire_type = 'medium' 
        if options and "initial_tire_type" in options:
            requested_tire = options.get("initial_tire_type", "medium")
            if requested_tire in self.tire_compound_properties:
                self.current_tire_type = requested_tire
        self._update_tire_specific_parameters()
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.rain_active = False
        self.rain_intensity = 0.0 
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.lap_log = []
        self.race_event_messages = [] 
        return self._get_obs(), {"message": "Environment reset", "initial_tire": self.current_tire_type}

    def _get_obs(self):
        return np.array([
            self.current_lap, self.tire_wear, self.traffic, self.fuel_weight,
            float(self.rain_active), float(self.safety_car_active), float(self.vsc_active)
        ], dtype=np.float32)

    def _update_tire_specific_parameters(self):
        props = self.tire_compound_properties.get(self.current_tire_type)
        if props:
            self.current_tire_wear_multiplier = props["wear_multiplier"]
            self.current_base_lap_time_offset = props["base_lap_time_offset"]
            self.current_tire_grip_bonus = props["grip_factor_bonus"]
            self.is_current_tire_rain_tire = props.get("is_rain_tire", False)
        else: 
            print(f"Warning: Unknown tire type '{self.current_tire_type}'. Using medium defaults.")
            self.current_tire_wear_multiplier = 1.0; self.current_base_lap_time_offset = 0.6;
            self.current_tire_grip_bonus = 0.0; self.is_current_tire_rain_tire = False;

    def step(self, action):
        if self.done: raise RuntimeError("Race has finished. Call reset().")
        self.current_lap += 1
        self._update_environmental_conditions() 
        self._handle_active_timed_events()    
        if not (self.safety_car_active or self.vsc_active): self._trigger_random_events()         

        pit_stop_time_loss_this_step = 0.0
        
        # Using simple formulaic lap time calculation
        lap_time = self.base_track_lap_time + self.current_base_lap_time_offset + \
                   (self.tire_wear / 100.0)**1.8 * 8 + \
                   (self.traffic * self.base_traffic_penalty_factor) + \
                   (self.fuel_weight * 0.032)
        
        if self.rain_active:
            if not self.is_current_tire_rain_tire: lap_time += 15 * self.rain_intensity
            lap_time += 5 * self.rain_intensity

        if self.safety_car_active: lap_time = max(lap_time, self.base_track_lap_time * 1.5)
        elif self.vsc_active: lap_time = max(lap_time, self.base_track_lap_time * 1.2)
        
        if action == 1: # Pit action
            pit_stop_time_loss_this_step = self.current_pit_time
            self.tire_wear = 0.0 
            self.race_event_messages.append(f"Lap {self.current_lap}: Agent decided to PIT.")
        else: # Stay out
            effective_wear_rate = self.current_tire_wear_multiplier * self.track_abrasiveness
            wear_increase_factor = 1.0 + ((1.0 - self.current_grip_factor) * 0.8)
            if self.rain_active and not self.is_current_tire_rain_tire: wear_increase_factor *= 3.0
            self.tire_wear = min(100, self.tire_wear + (effective_wear_rate * wear_increase_factor))
            
        final_lap_time = lap_time + pit_stop_time_loss_this_step
        self.fuel_weight = max(0, self.fuel_weight - random.uniform(1.6, 2.2))
        self.total_simulated_time += final_lap_time
        self._log_lap_data(final_lap_time, action)
        self.traffic = np.clip(np.random.normal(loc=0.3, scale=0.25), 0.0, 0.85)
        self.done = self.current_lap >= self.total_laps
        reward = -final_lap_time 
        info = { "lap_time": final_lap_time, "current_tire": self.current_tire_type, "tire_wear": self.tire_wear, "fuel": self.fuel_weight, "message": "Step successful"}
        return self._get_obs(), reward, self.done, False, info

    def _log_lap_data(self, lap_time, action_taken):
        lap_entry = { "lap": self.current_lap, "lap_time": round(float(lap_time), 3), "tire_wear": round(float(self.tire_wear), 2), "traffic": round(float(self.traffic), 3), "tire_type": self.current_tire_type, "action": int(action_taken), "fuel_weight": round(float(self.fuel_weight), 2), "track_temperature": round(float(self.track_temperature), 2), "grip_factor": round(float(self.current_grip_factor), 3), "rain": self.rain_active, "rain_intensity": round(self.rain_intensity, 2) if self.rain_active else 0.0, "safety_car_active": self.safety_car_active, "vsc_active": self.vsc_active }
        self.lap_log.append(lap_entry)
        log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "gym_race_lap_data.csv")
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        try:
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=lap_entry.keys());
                if write_header: writer.writeheader()
                writer.writerow(lap_entry)
        except IOError as e: print(f"Warning: Could not write to CSV log: {e}")

    def _update_environmental_conditions(self):
        self.track_temperature = np.clip(self.track_temperature + random.uniform(-0.2, 0.3), 15.0, 45.0)
        base_grip_change = random.uniform(0.0005, 0.0015)
        self.base_grip_factor = np.clip(self.base_grip_factor + base_grip_change, 0.8, 1.05)
        self.current_grip_factor = self.base_grip_factor + self.current_tire_grip_bonus
        if self.rain_active: self.current_grip_factor -= (0.4 * self.rain_intensity)
        self.current_grip_factor = np.clip(self.current_grip_factor, 0.35, 1.1)
        self.current_pit_time = self.base_pit_time * (1 + 0.3 * self.rain_intensity) if self.rain_active and self.rain_intensity > 0.1 else self.base_pit_time

    def _handle_active_timed_events(self):
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0: self.safety_car_active = False; self.race_event_messages.append(f"Lap {self.current_lap}: Safety Car IN THIS LAP.")
        if self.vsc_active:
            self.vsc_laps_remaining -= 1
            if self.vsc_laps_remaining <= 0: self.vsc_active = False; self.race_event_messages.append(f"Lap {self.current_lap}: VSC ENDING.")

    def _trigger_random_events(self):
        rand_val = random.random()
        if rand_val < 0.015: self.apply_safety_car(duration=random.randint(2,3))
        elif rand_val < 0.035: self.apply_vsc(duration=random.randint(1,2))
        elif rand_val < 0.045 and not self.rain_active : self.apply_rain(intensity=random.uniform(0.2, 0.5), is_forecasted=False)

    def apply_safety_car(self, duration=3):
        if not self.safety_car_active and not self.vsc_active:
            self.safety_car_active = True; self.safety_car_laps_remaining = duration
            self.race_event_messages.append(f"Lap {self.current_lap}: Safety Car DEPLOYED for {duration} laps.")

    def apply_vsc(self, duration=2):
        if not self.safety_car_active and not self.vsc_active:
            self.vsc_active = True; self.vsc_laps_remaining = duration
            self.race_event_messages.append(f"Lap {self.current_lap}: VSC DEPLOYED for {duration} laps.")

    def apply_rain(self, intensity=0.5, is_forecasted=True): 
        new_intensity = np.clip(intensity, 0.1, 1.0)
        if not self.rain_active:
            self.rain_active = True; self.rain_intensity = new_intensity; status = "Forecasted" if is_forecasted else "Unexpected"
            self.race_event_messages.append(f"Lap {self.current_lap}: {status} Rain Started (Intensity: {self.rain_intensity:.2f}).")
        elif abs(self.rain_intensity - new_intensity) > 0.1: 
            self.rain_intensity = new_intensity
            self.race_event_messages.append(f"Lap {self.current_lap}: Rain Intensity CHANGED to {self.rain_intensity:.2f}.")

    def clear_rain(self):
        if self.rain_active:
            self.rain_active = False; self.rain_intensity = 0.0
            self.race_event_messages.append(f"Lap {self.current_lap}: Rain has STOPPED.")

    def change_tire_type(self, tire_type_name):
        if tire_type_name in self.tire_compound_properties:
            if self.current_tire_type != tire_type_name:
                self.current_tire_type = tire_type_name; self._update_tire_specific_parameters()
                self.race_event_messages.append(f"Lap {self.current_lap}: Tires set to {tire_type_name}.")
        else:
            self.race_event_messages.append(f"Lap {self.current_lap}: Attempted to set UNKNOWN tire type: {tire_type_name}.")

    def render(self, mode='human'): 
        if mode == 'human':
            print(f"Lap: {self.current_lap}/{self.total_laps} | Tire: {self.current_tire_type} ({self.tire_wear:.1f}%) | Fuel: {self.fuel_weight:.1f}kg | Grip: {self.current_grip_factor:.2f} | Temp: {self.track_temperature:.1f}C | Rain: {self.rain_intensity:.1f if self.rain_active else 'No'}")
            if self.safety_car_active: print(f"SAFETY CAR ACTIVE ({self.safety_car_laps_remaining} laps left)")
            if self.vsc_active: print(f"VSC ACTIVE ({self.vsc_laps_remaining} laps left)")
        pass

    def close(self):
        pass

if __name__ == '__main__':
    env = PitStopEnv(total_laps=20, base_lap_time_seconds=75.0) 
    obs, info = env.reset()
    done = False
    for i in range(env.total_laps + 2):
        action = env.action_space.sample() 
        if env.done: break
        print(f"\n--- Simulating Lap {env.current_lap + 1}, Action: {'Pit' if action ==1 else 'Stay Out'} ---")
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    print("\nRace Event Messages:", env.race_event_messages)