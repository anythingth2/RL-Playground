# %%
import numpy as np

import gym
# %%
n_level = 4
q_table = np.random.random((n_level, 2))

env = gym.make('CartPole-v0')

OBSERVATION_MAX = 4e1
OBSERVATION_MIN = -4e-1
observation_low = env.observation_space.low.clip(min=OBSERVATION_MIN)
observation_high = env.observation_space.high.clip(max=OBSERVATION_MAX)


for i_episode in range(20):
    observation = env.reset()
    state = 0
    for t in range(100):
        env.render()
        # print(q_table)
        # print('-'*22)
        action = q_table[state].argmax()

        observation, reward, done, info = env.step(action)

        next_state = observation.clip(min=OBSERVATION_MIN, max=OBSERVATION_MAX)
        next_state = (next_state - observation_low) / \
            (observation_high - observation_low)
        next_state = int(next_state[0] * n_level * n_level // n_level)
        
        # print(f'select action:{action} state:{state} next_state:{next_state}')
        print(f'reward {reward} {info}')

        q_table[state, action] = -reward + q_table[next_state].max()

        state = next_state
        if done:
            print(f"{done}  Episode finished after {t+1} timesteps")
            break
env.close()

