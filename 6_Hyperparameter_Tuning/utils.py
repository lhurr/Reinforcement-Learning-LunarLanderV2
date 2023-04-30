import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, LeakyReLU, ReLU, Conv2D
from tensorflow.keras.models import load_model, Model, model_from_json

from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import NoisyDense
from tensorflow.keras import backend as K
import wandb
import numpy as np
import pandas as pd

class NoisyD3QN(Model):
    def __init__(self, num_actions, fc1, fc2):
        super(NoisyD3QN, self).__init__()
        # self.dense1 = Dense(fc1, activation='relu')
        # self.dense2 = Dense(fc2, activation='relu')
        self.dense1 = Dense(fc1, activation='relu')
        self.dense2 = Dense(fc2, activation='relu')
        self.V = NoisyDense(1)
        self.A = NoisyDense(num_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        avg_A = tf.math.reduce_mean(A, axis=1, keepdims=True)
        Q = (V + (A - avg_A))
        return Q, A

class ReplayBuffer:
    def __init__(self, size, input_shape):
        self.size = size
        self.counter = 0
        self.state_buffer = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.new_state_buffer = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.terminal_buffer = np.zeros(self.size, dtype=np.bool_)

    def store_tuples(self, state, action, reward, new_state, done):
        idx = self.counter % self.size
        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.new_state_buffer[idx] = new_state
        self.terminal_buffer[idx] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_buffer = min(self.counter, self.size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        new_state_batch = self.new_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch
class NoisyNet:
    def __init__(self, lr, discount_factor, num_actions, epsilon, batch_size, input_dim, update_rate):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.update_rate = update_rate
        self.step_counter = 0
        self.buffer = ReplayBuffer(100000, input_dim)
        self.q_net = NoisyD3QN(num_actions, 128, 128)
        self.q_target_net = NoisyD3QN(num_actions, 128, 128)
        self.q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.q_target_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def policy(self,observation):
        state = np.array([observation])
        _, actions = self.q_net(state)
        action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def train(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_rate == 1:
            # print(self.step_counter)
            self.q_target_net.set_weights(self.q_net.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted, _ = self.q_net(state_batch)
        q_next, _ = self.q_target_net(new_state_batch)
        q_target = q_predicted.numpy()
        _, actions = self.q_net(new_state_batch)
        max_actions = tf.math.argmax(actions, axis=1)

        for idx in range(done_batch.shape[0]):
            q_target[idx, action_batch[idx]] = reward_batch[idx] + self.discount_factor*q_next[idx, max_actions[idx]] *\
                                               (1-int(done_batch[idx]))

        self.q_net.train_on_batch(state_batch, q_target)
        self.step_counter += 1
    def train_model(self, env, num_episodes, graph):

        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        f = 0
        txt = open("tuned_d3qn.txt", "w")

        for i in range(num_episodes):
            done = False
            score = 0.0
            state = env.reset()
            while not done:
                action = self.policy(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, new_state, done)
                state = new_state
                self.train()
            scores.append(score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
                                                                             avg_score))
            wandb.log(
                {
                    "Episode": i,
                    "Reward": score,
                    "Avg-Reward-100e": avg_score,
                }
            )
            if avg_score >= 200.0 and score >= 250:
                self.q_net.save(("tuned_d3qn/tuned_d3qn{0}".format(f)))
                self.q_net.save_weights(("tuned_d3qn/tuned_d3qn{0}/net_weights{0}.h5".format(f)))
                txt.write("Saved {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(f, i, num_episodes,
                                                                                                  score, self.epsilon,
                                                                                                  avg_score))
                f += 1

        txt.close()

        df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})
        return df

    def test(self, env, num_episodes, file_type, file, graph):
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.q_net.load_weights(file)
        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                env.render()
                action = self.policy(state)
                new_state, reward, done, _ = env.step(action)
                episode_score += reward
                state = new_state
            score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        # if graph:
        #     df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})
        df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})
        env.close()
        return df

        