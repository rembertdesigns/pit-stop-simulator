import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PitStopEnv(gym.Env):
    def __init__(self, total_laps=58):
        super().__init__()
        self.total_laps = total_laps
        self.max_tire_wear = 100.0
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = 0.0

        # Observation space: [current_lap, tire_wear, traffic]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.total_laps, self.max_tire_wear, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: 0 = stay out, 1 = pit
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

        # Apply action
        if action == 1:  # pit
            self.tire_wear = 0.0
            lap_time = 30  # pit cost
            reward = -lap_time
            reward += 5  # small bonus to encourage pitting
        else:  # stay out
            self.tire_wear += 1.5
            degradation = self.tire_wear * 0.3
            traffic_penalty = self.traffic * 5
            lap_time = 20 + degradation + traffic_penalty
            reward = -lap_time

            # Penalize worn tires
            if self.tire_wear > 70:
                reward -= 20
            elif self.tire_wear > 50:
                reward -= 10

        self.current_lap += 1
        self.traffic = np.random.rand()

        if self.current_lap >= self.total_laps:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.current_lap, self.tire_wear, self.traffic], dtype=np.float32)

