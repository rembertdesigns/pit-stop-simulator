import numpy as np

class QAgent:
    def __init__(self, env,  # Pass the actual environment instance
                 buckets=(58, 10, 10, 2, 2, 2), # (laps, tire_wear, traffic, rain, sc, vsc)
                 alpha=0.1, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        
        self.env = env # Store the environment instance

        # Initialize attributes first to ensure they always exist on the instance
        self.env_observation_space_shape = None 
        self.env_action_space_n = None

        if hasattr(self.env, 'observation_space') and hasattr(self.env.observation_space, 'shape'):
            self.env_observation_space_shape = self.env.observation_space.shape
        else:
            print("WARNING (QAgent __init__): env.observation_space.shape not found. Assuming default shape (7,).")
            self.env_observation_space_shape = (7,) # Default based on PitStopEnv's 7-component observation

        if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'n'):
            self.env_action_space_n = self.env.action_space.n
        else:
            print("WARNING (QAgent __init__): env.action_space.n not found. Assuming 2 actions.")
            self.env_action_space_n = 2 
            
        self.buckets = buckets
        if len(self.buckets) != 6: 
            raise ValueError(f"Buckets tuple length ({len(self.buckets)}) must be 6 for the defined state (laps, wear, traffic, rain, sc, vsc).")
            
        self.q_table = np.zeros(self.buckets + (self.env_action_space_n,))
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def set_env(self, env):
        """
        Allows re-associating the agent with an environment instance,
        crucial if the agent is loaded from a file.
        """
        self.env = env 
        
        # Re-initialize attributes based on the new env
        self.env_observation_space_shape = None
        self.env_action_space_n = None

        if hasattr(self.env, 'observation_space') and hasattr(self.env.observation_space, 'shape'):
            self.env_observation_space_shape = self.env.observation_space.shape
        else:
            print("WARNING (QAgent set_env): env.observation_space.shape not found. Assuming default shape (7,).")
            self.env_observation_space_shape = (7,)
            
        if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'n'):
            self.env_action_space_n = self.env.action_space.n
        else:
            print("WARNING (QAgent set_env): env.action_space.n not found. Assuming 2 actions.")
            self.env_action_space_n = 2

    def discretize(self, obs):
        if isinstance(obs, tuple) and len(obs) > 0 and isinstance(obs[0], np.ndarray):
            current_obs_array = obs[0]
        elif isinstance(obs, np.ndarray):
            current_obs_array = obs
        else:
            raise TypeError(f"Observation must be a NumPy array or a tuple (array, info_dict), got {type(obs)}")

        # Check if the attribute exists AND is not None before trying to access its elements
        if not hasattr(self, 'env_observation_space_shape') or self.env_observation_space_shape is None:
             # This is the explicit raise you added, confirming the attribute isn't properly set.
             # The __init__ or set_env should prevent this if they run correctly.
             raise AttributeError("QAgent instance is missing or has None for 'env_observation_space_shape'. Was it initialized/set_env called correctly with an environment?")

        if len(current_obs_array) != self.env_observation_space_shape[0]:
             raise ValueError(f"Observation length mismatch. Expected {self.env_observation_space_shape[0]} (from {self.env_observation_space_shape}), got {len(current_obs_array)}")

        lap, tire_wear, traffic, _fuel_weight, rain_active, sc_active, vsc_active = current_obs_array[:7]
        
        lap_bin = min(int(lap), self.buckets[0] - 1); lap_bin = max(0, lap_bin)
        tire_bin = min(int(tire_wear / (100.0 / self.buckets[1])), self.buckets[1] - 1); tire_bin = max(0, tire_bin)
        traffic_bin = min(int(traffic * self.buckets[2]), self.buckets[2] - 1); traffic_bin = max(0, traffic_bin)
        rain_bin = min(int(rain_active), self.buckets[3] - 1); rain_bin = max(0, rain_bin)
        sc_bin = min(int(sc_active), self.buckets[4] - 1); sc_bin = max(0, sc_bin)
        vsc_bin = min(int(vsc_active), self.buckets[5] - 1); vsc_bin = max(0, vsc_bin)
        
        return (lap_bin, tire_bin, traffic_bin, rain_bin, sc_bin, vsc_bin)

    def choose_action(self, state_tuple):
        if not isinstance(state_tuple, tuple) or len(state_tuple) != len(self.buckets):
            raise ValueError(f"Invalid state_tuple for choose_action. Expected tuple of length {len(self.buckets)}.")
        if np.random.random() < self.epsilon:
            if self.env_action_space_n is None: # Should not happen if __init__ or set_env was called
                raise ValueError("QAgent's env_action_space_n is None.")
            return np.random.randint(0, self.env_action_space_n)
        else:
            return np.argmax(self.q_table[state_tuple])

    def update_q(self, state_tuple, action, reward, next_state_tuple):
        if not isinstance(state_tuple, tuple) or len(state_tuple) != len(self.buckets):
            raise ValueError(f"Invalid state_tuple for update_q. Expected tuple of length {len(self.buckets)}.")
        if not isinstance(next_state_tuple, tuple) or len(next_state_tuple) != len(self.buckets):
            raise ValueError(f"Invalid next_state_tuple for update_q. Expected tuple of length {len(self.buckets)}.")
        if self.env_action_space_n is None or not (0 <= action < self.env_action_space_n):
             raise ValueError(f"Invalid action {action}. Must be between 0 and { (self.env_action_space_n -1) if self.env_action_space_n else 'N/A'}.")

        best_future_q = np.max(self.q_table[next_state_tuple])
        current_q_value = self.q_table[state_tuple + (action,)]
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_future_q - current_q_value)
        self.q_table[state_tuple + (action,)] = new_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)