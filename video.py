import gymnasium as gym

env =  gym.make(id='Hopper-v4', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="./video/")

st, info = env.reset(seed=43)

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

