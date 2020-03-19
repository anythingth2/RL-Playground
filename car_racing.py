#%%
import gym
import numpy as np
from PIL import Image
# %%
# Action space: steer, gas, brake


env = gym.make('CarRacing-v0')
# %%
observation = env.reset()



# %%
Image.fromarray(observation)

# %%
env.action_space.low

# %%


# %%
