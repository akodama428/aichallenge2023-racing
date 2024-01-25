from .env import CustomEnv
import gymnasium as gym

gym.register(
    id='myenv-v0',
    entry_point='myenv.env:CustomEnv',
    max_episode_steps=2000,
)