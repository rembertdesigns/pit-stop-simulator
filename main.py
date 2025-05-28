from env.gym_race_env import PitStopEnv

def test_environment():
    env = PitStopEnv()
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("🏁 Starting race simulation...")

    while not done:
        action = env.action_space.sample()  # Random: 0 = stay out, 1 = pit
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Lap {int(obs[0])} | Action: {'PIT' if action else 'STAY OUT'} | "
              f"Tire Wear: {obs[1]:.2f} | Traffic: {obs[2]:.2f} | Reward: {reward:.2f}")
        step += 1

    print(f"\n✅ Simulation complete in {step} laps. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_environment()
