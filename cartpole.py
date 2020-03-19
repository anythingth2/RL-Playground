# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gym
# %%
n_level = 12
q_shape = (n_level, )*2 + (2,)
# q_shape = (n_level, 2)
# q_table = np.random.random(q_shape)
q_table = np.zeros(q_shape)
discount_factor = 0.9
lr = 0.9
env = gym.make('CartPole-v1')


observation_low = env.observation_space.low.copy()
observation_low[[1, 3]] = -4
observation_high = env.observation_space.high.copy()
observation_high[[1, 3]] = 4


def norm_observation(observation):
    observation = observation.clip(min=observation_low, max=observation_high)
    observation = (observation - observation_low) / \
        (observation_high - observation_low)
    return observation


def encode_state(observation):
    observation = norm_observation(observation)
    # print(observation,)
    # observation = observation.mean()
    # state = int(observation[0] * n_level * n_level // n_level)
    state = (observation * n_level * n_level // n_level).astype(int)
    state = state[2:]
    return state


def select_action(state):
    q_selected = q_table[tuple(state)]
    action = q_selected.argmax()

    return action


def update_Q(state, next_state):
    value = lr*(reward +
                discount_factor * q_table[tuple(next_state)].max() - q_table[tuple(state)][action])

    q_table[tuple(state)][action] =  q_table[tuple(state)][action] + value
    # print(f'update {state} with {value} | {q_table[tuple(state)]}')
#%%

for i_episode in range( 100):
    observation = env.reset()
    state = encode_state(observation)
    for t in range(1000):
        env.render()
        if i_episode < 20:
            action = env.action_space.sample()
        else:
            action = select_action(state)

        observation, reward, done, info = env.step(action)
        next_state = encode_state(observation)
        # print(
        # f'select action:{action} state:{state} next_state:{next_state} reward:{reward}')

        # print(f'action: {action}', end=' ')
        update_Q(state, next_state)
        state = next_state
        if done:
            print(
                f"{done}  Episode {i_episode} finished after {t+1} timesteps {info}")
            break
env.close()


# %%
sns.heatmap(q_table[:, :, 0])
# %%
sns.heatmap(q_table[:, :, 1])

# %%
sns.heatmap(q_table.mean(axis=-1))


# %%
