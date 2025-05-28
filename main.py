from env.gym_race_env import PitStopEnv
from rl.q_learning_agent import QAgent

def train_q_learning(episodes=500):
    env = PitStopEnv()
    agent = QAgent(env)
    rewards = []

    for episode in range(episodes):
        obs = env.reset()
        state = agent.discretize(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = agent.discretize(next_obs)
            agent.update_q(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    return rewards

if __name__ == "__main__":
    train_q_learning()

