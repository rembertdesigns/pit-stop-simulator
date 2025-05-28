import gym
import numpy as np
from gym import spaces

class PitStopEnv(gym.Env):
    def __init__(self, total_laps=58):
        super(PitStopEnv, self).__init__()
        self.total_laps = total_laps
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = 0.0

        # Define observation: [current_lap, tire_wear, traffic]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0]),
            high=np.array([self.total_laps, 100.0, 1.0]),
            dtype=np.float32
        )

        # Action: 0 = stay out, 1 = pit
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_lap = 0
        self.tire_wear = 0.0
        self.traffic = np.random.rand()  # [0, 1]
        return self._get_obs()

    def step(self, action):
        done = False
        reward = 0

        # Apply action
        if action == 1:  # pit
            self.tire_wear = 0.0
            lap_time = 30  # pit cost
        else:  # stay out
            self.tire_wear += 1.5
            degradation = self.tire_wear * 0.3
            traffic_penalty = self.traffic * 5
            lap_time = 20 + degradation + traffic_penalty

        # Reward = negative lap time (we want it shorter)
        reward = -lap_time

        self.current_lap += 1
        self.traffic = np.random.rand()

        if self.current_lap >= self.total_laps:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([self.current_lap, self.tire_wear, self.traffic], dtype=np.float32)
