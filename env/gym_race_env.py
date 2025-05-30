import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PitStopEnv(gym.Env):
    def __init__(self, total_laps=58, tire_degradation=0.3, traffic_penalty=5.0, pit_time=30.0, tire_wear_rate=1.5):
        super().__init__()
        self.total_laps = total_laps
        self.tire_degradation = tire_degradation
        self.traffic_penalty = traffic_penalty
        self.pit_time = pit_time
        self.tire_wear_rate = tire_wear_rate  # ✅ store it

        self.max_tire_wear = 100.0
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = 0.0

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.total_laps, self.max_tire_wear, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = np.random.rand()
        return self._get_obs(), {}

    def step(self, action):
        terminated = False
        truncated = False

        if action == 1:  # Pit
            self.tire_wear = 0.0
            lap_time = self.pit_time
            reward = -lap_time + 3  # Small incentive for pitting
        else:
            self.tire_wear += self.tire_wear_rate
            degradation = self.tire_wear * self.tire_degradation
            traffic_cost = self.traffic * self.traffic_penalty
            lap_time = 20 + degradation + traffic_cost

            reward = -lap_time
            if self.tire_wear > 50:
                reward -= 10
            elif self.tire_wear > 70:
                reward -= 20

        self.current_lap += 1
        self.traffic = np.random.rand()

        if self.current_lap >= self.total_laps:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.current_lap, self.tire_wear, self.traffic], dtype=np.float32)