import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from collections import deque
from environment.Parallel_Machine import JobShop

class DDQN(tf.keras.Model):
    def __init__(self, a_size):
        super().__init__(name='ddqn')
        self.hidden1 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.hidden2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.hidden3 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")
        self.out = tf.keras.layers.Dense(a_size)

    def call(self, inputs):
        hidden1 = self.hidden1(inputs)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        q_values = self.out(hidden3)
        return q_values


class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model_path = '../model/ddqn/queue-%d' % action_size

        self.model = DDQN(self.action_size)
        self.model.load_weights(self.model_path)

    def get_action(self, state, dispatching):
            q_value = self.model(state)
            if dispatching:
                return np.argmax(q_value[0][:-1])
            else:
                return len(q_value[0]) - 1


if __name__ == "__main__":

    state_size = 95
    action_size = 5

    agent = DDQNAgent(state_size=state_size, action_size=action_size)

    env = JobShop()

    num_episode = 100
    mean_weighted_tardiness_list = []
    for e in range(num_episode):
        np.random.seed(e)
        done = False
        epsisode_reward = 0

        state, next_dispatching = env.reset()
        state = np.reshape(state, [1, state_size])

        action_list = []

        while not done:
            # RL agent 를 이용할 때에는 아래 세 줄을 사용
            action = agent.get_action(state, next_dispatching)
            action_list.append(action)
            next_state, reward, done, next_dispatching = env.step(action)
            # Heuristic dispatching rule 을 쓸 때는 아래 한 줄을 사용
            # next_state, reward, done, _ = env.step(3)

            epsisode_reward += reward

            state = next_state
            state = np.reshape(state, [1, state_size])

            if done:
                mean_weighted_tardiness = env.mean_weighted_tardiness
                mean_weighted_tardiness_list.append(mean_weighted_tardiness)
                print("episode: {:3d} | episode reward: {:5.4f} | mean weighted tardiness: {:5.4f}".format(e, epsisode_reward, mean_weighted_tardiness))
                print(action_list)

    mean_weighted_tardiness_list = np.array(mean_weighted_tardiness_list)
    avg_mean_weighted_tardiness = np.average(mean_weighted_tardiness_list)
    std_mean_weighred_tardiness = np.std(mean_weighted_tardiness_list)
    print('Average mean weighted tardiness : ', avg_mean_weighted_tardiness)
    print('Std mean weighted tardiness : ', std_mean_weighred_tardiness)
    print(mean_weighted_tardiness_list)

