import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env, rand_act_prob=1.0):
        self.env     = env
        self.memory  = deque(maxlen=10000)
        
        self.discount_factor    = 0.95       # discount
        self.epsilon            = rand_act_prob
        self.epsilon_min        = 0.01

        self.trials_2_decay     = 200

        self.steps_2_learn_skip = 600

        self.epsilon_decay      = (self.epsilon - self.epsilon_min) / (self.trials_2_decay * 200)
        self.learning_rate      = 1e-3
        self.tau = .125

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 64

        if len(self.memory) < self.steps_2_learn_skip: 
            return

        samples = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for idx, sample in enumerate(samples):
            state, action, reward, new_state, done = sample
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + \
                                 np.amax(self.target_model.predict(new_state)[0]) * self.discount_factor

            update_input[idx] = state
            update_target[idx] = target

        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()

        # target_weights = self.target_model.get_weights()
        
        # for i in range(len(target_weights)):
            # target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        # self.target_model.set_weights(target_weights)
        self.target_model.set_weights(weights)

    def save_model(self, fn):
        self.model.save_weights(fn)

    def load_model(self, fn):
        self.model.load_weights(fn)
