import numpy as np

class QAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)  # pit or not
        return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (reward + self.gamma * best_next - self.q_table[state, action])
