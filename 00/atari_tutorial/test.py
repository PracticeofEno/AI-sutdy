import gymnasium as gym

env = gym.make("CarRacing-v2", domain_randomize=True)
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample())
env.close()  # https://github.com/openai/gym/issues/893