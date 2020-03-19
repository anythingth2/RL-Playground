# %%
from collections import namedtuple
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras import Sequential
from keras.optimizers import Adam
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
# %%
Transition = namedtuple(
    'Transition', ['state', 'action', 'reward', 'next_state'])

# %%


class DQNAgent:

    def __init__(self,
                 input_shape=(96, 96, 3),
                 action_space_high=(1, 1, 1),
                 action_space_low=(-1, 0, 0),
                 action_space_len=3,
                 discretize_level=8,
                 memory_size=128):
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.discretize_level = discretize_level
        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0

        self.policy_model = self.build_model(
            input_shape, discretize_level**action_space_len)
        self.target_model = self.build_model(
            input_shape, discretize_level**action_space_len)
        self.target_model.set_weights(self.policy_model.get_weights())

    def push_memory(self, transition: Transition):
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[self.memory_index] = transition
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def build_model(self, input_shape: tuple, output_len: int, ) -> Sequential:
        model = Sequential([
            Conv2D(64, kernel_size=3, padding='same',
                   activation='relu', input_shape=input_shape),
            Flatten(),
            Dense(output_len, activation='linear')
        ])
        model.compile(optimizer=Adam(), loss='mse', )
        return model

    def encode_action_index(self, action, flatten=False):
        def encode(action_value, high, low):
            action_value = (action_value - low) / (high - low)
            idx = int(action_value * self.discretize_level *
                      self.discretize_level // self.discretize_level)
            return max(min(idx, self.discretize_level-1), 0)
        action_idxes = [encode(action_value, high, low) for action_value, high, low in zip(
            action, self.action_space_high, self.action_space_low)]
        if flatten:
            return np.ravel_multi_index(action_idxes, (self.discretize_level,)*len(action_idxes))
        else:
            return action_idxes

    def decode_action_index(self, action_idx):
        def decode(idx, high, low):
            space = np.linspace(low, high, self.discretize_level+1)
            upper_bounds = space[1:]
            lower_bounds = space[:-1]
            center_bounds = (upper_bounds + lower_bounds) / 2
            return center_bounds[idx]
        idxes = np.unravel_index(action_idx, shape=(
            self.discretize_level,)*len(self.action_space_high))
        return [decode(idx, high, low) for idx, high, low in zip(idxes, self.action_space_high, self.action_space_low)]

    def select_action(self, state):
        q = self.policy_model.predict(np.array([state]))[0]
        action_idx = q.argmax()
        return self.decode_action_index(action_idx)

    def train(self, batch_size=32):
        discount_value = 0.9

        batch_idxes = np.random.choice(
            len(self.memory), size=min(batch_size, len(self.memory)), replace=False)
        batch: List[Transition] = [self.memory[idx] for idx in batch_idxes]

        current_states = np.array([transition.state for transition in batch])
        target_qs = self.policy_model.predict(current_states)

        next_states = np.array([transition.next_state for transition in batch])
        next_qs = self.target_model.predict(next_states)
        ys = []
        for i, (state, action, reward, next_state) in enumerate(batch):
            target_q = target_qs[i]
            next_q = next_qs[i]
            target_q[self.encode_action_index(
                action, flatten=True)] = reward + discount_value * np.max(next_q)

        self.policy_model.fit(current_states, target_qs,
                              batch_size=batch_size, )


# %%
agent = DQNAgent()
# %%
env = gym.make('CarRacing-v0')
for episode_idx in range(100):
    observation = env.reset()
    timestep = 0
    while True:
        if episode_idx < 20:
            action = env.action_space.sample()
        else:
            action = agent.select_action(observation)

        next_observation, reward, done, info = env.step(action)

        if reward > 0:
            agent.push_memory(Transition(
                state=observation, action=action, reward=reward, next_state=next_observation))
            agent.train()

            observation = next_observation
            timestep += 1
            print(f'Timestep {timestep} action:{action} reward:{reward}')

        if done:
            print(f'Done with {timestep} timestep')
            break

        env.render()

env.close()
# %%
