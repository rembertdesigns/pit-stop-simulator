import simpy
import numpy as np

class F1RaceEnv:
    def __init__(self, env, total_laps=58):
        self.env = env
        self.total_laps = total_laps
        self.current_lap = 0
        self.tire_wear = 0.0
        self.pitted = False

    def run_lap(self):
        while self.current_lap < self.total_laps:
            lap_time = self.base_lap_time() + self.tire_wear_penalty()
            yield self.env.timeout(lap_time)
            self.tire_wear += 1.5  # increase wear
            self.current_lap += 1

    def pit_stop(self):
        yield self.env.timeout(20)  # pit stop delay
        self.tire_wear = 0
