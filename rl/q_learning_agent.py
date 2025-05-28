import numpy as np

class QAgent:
    def __init__(self, env, buckets=(58, 10, 10), alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.buckets = buckets  # Discretization bins: (laps, tire_wear, traffic)
        self.q_table = np.zeros(self.buckets + (env.action_space.n,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

    def discretize(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]  # ✅ Properly unpack obs if it's a (obs, info) tuple
        lap, tire, traffic = obs
        lap_bin = min(int(lap), self.buckets[0] - 1)
        tire_bin = min(int(tire // 10), self.buckets[1] - 1)
        traffic_bin = min(int(traffic * 10), self.buckets[2] - 1)
        return (lap_bin, tire_bin, traffic_bin)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        best_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = current_q + self.alpha * (reward + self.gamma * best_future_q - current_q)
        self.q_table[state + (action,)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        best_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = current_q + self.alpha * (reward + self.gamma * best_future_q - current_q)
        self.q_table[state + (action,)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)