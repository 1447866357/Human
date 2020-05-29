from BaseQLearning import *
import tensorflow as tf
import numpy as np
import random


class ControlSharingAgent:
    def __init__(self, env, base_model_path, model_path, L, R, C):
        self.teacher = BaseDeepQLearning(env, base_model_path)
        self.original = [L, R, C, 1]

        with tf.variable_scope('ControlSharing'):
            self._create_DQN()
            self._create_train()

        self.replay = deque()
        self.sess = tf.Session()
        self.model_path = model_path
        self.saver = tf.train.Saver()
        self.init_network()


    def init_network(self):
        self.L, self.R, self.C, self.B = self.original
        with tf.variable_scope('ControlSharing'):
            self.sess.run(tf.initialize_all_variables())

    def _create_DQN(self):
        self.state_ph = tf.placeholder(tf.float32, [None, 4])
        W1 = self._weight_variable([4, 20])
        b1 = self._bias_variable([20])
        W2 = self._weight_variable([20, 2])
        b2 = self._bias_variable([2])
        hidden_layer = tf.nn.tanh(tf.matmul(self.state_ph, W1) + b1)
        self.Q = tf.matmul(hidden_layer, W2) + b2

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def _create_train(self):
        self.y_ph = tf.placeholder(tf.float32, [None])
        self.action_ph = tf.placeholder(tf.float32, [None, 2])
        batch_Q = tf.reduce_sum(self.Q * self.action_ph, reduction_indices=1)
        loss = tf.reduce_mean(tf.square(batch_Q - self.y_ph))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    def egreedy_action(self, state, epsilon):
        if random.random() < self.B * self.L:
            return self.teacher.egreedy_action(state, 1-self.C)

        if random.random() < epsilon:
            return 1 if random.random() > 0.5 else 0

        return self.greedy_action(state)

    def greedy_action(self, state):
        with tf.variable_scope('ControlSharing'):
            q = self.sess.run(self.Q, feed_dict={self.state_ph: [state]})[0]
            return np.argmax(q)

    def dump(self):
        self.saver.save(self.sess, self.model_path)

    def update(self, state, action, reward, new_state, done):
        y = 0
        if done:
            y = reward
        else:
            y = reward + 0.99 * np.max(self.sess.run(self.Q, feed_dict={self.state_ph:[new_state]})[0])

        a = [0, 0]
        a[action] = 1
        self.replay.append((state, a, y))
        if len(self.replay) > REPLAY_SIZE:
            self.replay.popleft()
        if len(self.replay) > BATCH_SIZE:
            self.train()

    def train(self):
        with tf.variable_scope('ControlSharing'):
            batch = random.sample(self.replay, BATCH_SIZE)
            states = [data[0] for data in batch]
            actions = [data[1] for data in batch]
            ys = [data[2] for data in batch]
            self.sess.run(self.optimizer, feed_dict = {
                self.y_ph : ys,
                self.action_ph : actions,
                self.state_ph : states
            })

    def close(self):
        self.sess.close()
        self.teacher.close()
        tf.reset_default_graph()
