from BaseQLearning import *
import tensorflow as tf
import numpy as np
import random


class MultiBanditAgent:
    def __init__(self, env, base_model_path, model_path, L, R, C):
        self.teacher = BaseDeepQLearning(env, base_model_path)
        self.original = [L, R, C, 1]
        self.L, self.R, self.C, self.B = self.original

        with tf.variable_scope('MultiBandit'):
            self._create_DQN()
            self._create_train()

        self.methods_weight = np.zeros(4)
        self.change_method = True
        self.methods_weight_dump = []
        self.methods = [self.action_biasing_egreedy_action,
                        self.reward_shaping_egreedy_action,
                        self.control_sharing_egreedy_action,
                        self.q_update_egreedy_action]

        self.importance_methods = [self.action_biasing_egreedy_prob,
                                   self.reward_shaping_egreedy_prob,
                                   self.contorl_sharing_egreedy_prob,
                                   self.q_update_egreedy_prob]
        self.method_gamma = 0.1
        self.traj = []
        self.importance = np.ones(4)

        self.replay = deque()
        self.sess = tf.Session()
        self.model_path = model_path
        self.saver = tf.train.Saver()
        self.current_used = self.get_best_method_by_prob()
        self.change_method = False
        self.init_network()

    def init_network(self):
        self.L, self.R, self.C, self.B = self.original
        self.methods_weight = np.zeros(4)
        self.methods_weight_dump = []
        self.current_used = self.get_best_method_by_prob()
        self.importance = np.ones(4)
        self.change_method = False
        self.methods_weight_dump = []
        self.method_gamma = 0.1
        self.traj = []
        with tf.variable_scope('MultiBandit'):
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

    def method_weight_update(self, reward):
        reward = reward / 200
        for method_index in range(len(self.methods)):
            factor = self.importance[method_index]
            if factor > 2:
                factor = 2
            self.methods_weight[method_index] += self.method_gamma * (reward * factor - self.methods_weight[method_index])

# four method
    def reward_shaping_egreedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return 1 if random.random() > 0.5 else 0

        return self.greedy_action(state)

    def q_update_egreedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return 1 if random.random() > 0.5 else 0

        return self.greedy_action(state)

    def control_sharing_egreedy_action(self, state, epsilon):
        if random.random() < self.B * self.L:
            return self.teacher.egreedy_action(state, 1-self.C)

        if random.random() < epsilon:
            return 1 if random.random() > 0.5 else 0

        return self.greedy_action(state)

    def action_biasing_egreedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return 1 if random.random() > 0.5 else 0
        with tf.variable_scope('MultiBandit'):
            teacher_action = self.teacher.egreedy_action(state, 1-self.C)
            human_reward = np.zeros(2)
            if random.random() < self.L:
                human_reward[teacher_action] = self.R
            q = self.sess.run(self.Q, feed_dict={self.state_ph: [state]})[0]
            # print(q + self.B * human_reward)
            return np.argmax(q + self.B * human_reward)
# end

    def get_current_method_touse(self):
        if not self.change_method:
            return self.current_used
        else:
            self.change_method = False
            return self.get_best_method_by_prob()

    def get_best_method_by_prob(self):
        prob = np.exp(5*self.methods_weight[:4]-self.methods_weight.min())
        # print(self.methods_weight[:3])
        prob /= np.sum(prob)
        return np.random.choice(len(prob), 1, p=prob)[0]

    def action_biasing_egreedy_prob(self, state, action, epsilon):
        with tf.variable_scope('MultiBandit'):
            teacher_greedy_action = self.teacher.greedy_action(state)
            human_best_reward = np.zeros(2)
            human_worst_reward = np.zeros(2)
            human_best_reward[teacher_greedy_action] = self.R
            human_worst_reward[1 - teacher_greedy_action] = self.R
            q = self.sess.run(self.Q, feed_dict={self.state_ph: [state]})[0]
            # print(q + self.B * human_reward)
            best_action = np.argmax(q + self.B * human_best_reward)
            worst_action = np.argmax(q + self.B * human_worst_reward)
        action_prob = [0, 0]
        action_prob[best_action] += epsilon/4 + (1 - epsilon/2) * self.C
        action_prob[1 - best_action] += epsilon/4

        action_prob[worst_action] += epsilon/4 + (1 - epsilon/2) * (1 - self.C)
        action_prob[1 - worst_action] += epsilon/4

        return action_prob[action]

    def reward_shaping_egreedy_prob(self, state, action, epsilon):
        best_action = self.greedy_action(state)

        action_prob = [0, 0]
        action_prob[best_action] = 1 - epsilon/2
        action_prob[1 - best_action] = epsilon/2

        return action_prob[action]

    def q_update_egreedy_prob(self, state, action, epsilon):
        best_action = self.greedy_action(state)

        action_prob = [0, 0]
        action_prob[best_action] = 1 - epsilon/2
        action_prob[1 - best_action] = epsilon/2

        return action_prob[action]


    def contorl_sharing_egreedy_prob(self, state, action, epsilon):
        teacher_best_action = self.teacher.greedy_action(state)
        teacher_worst_action = 1 - teacher_best_action
        student_best_action = self.greedy_action(state)
        student_worst_action = 1 - student_best_action

        action_prob = [0, 0]
        action_prob[teacher_best_action] += self.L * self.C
        action_prob[teacher_worst_action] += self.L * (1 - self.C)
        action_prob[student_best_action] += (1 - self.L) * (1 - epsilon/2)
        action_prob[student_worst_action] += (1 - self.L) * epsilon/2

        return action_prob[action]

    def egreedy_action(self, state, epsilon):
        current_action = self.methods[self.current_used](state, epsilon)
        current_prob = self.importance_methods[self.current_used](state, current_action, epsilon)
        for method_index in range(len(self.methods)):
            self.importance[method_index] *= self.importance_methods[method_index](state, current_action, epsilon) / current_prob
        return current_action

    def greedy_action(self, state):
        with tf.variable_scope('MultiBandit'):
            q = self.sess.run(self.Q, feed_dict={self.state_ph: [state]})[0]
            return np.argmax(q)

    def dump(self):
        self.saver.save(self.sess, self.model_path)
        import pickle
        pickle.dump(self.methods_weight_dump, open('model/feedback/multi_bandit/weight.pkl', 'wb'))

    def update(self, state, action, reward, new_state, done):
        y = 0
        if done:
            self.methods_weight_dump.append(self.methods_weight.copy())
            self.method_weight_update(reward)
            self.change_method = True
            self.importance = np.ones(4)
            self.current_used = self.get_current_method_touse()
            y = reward
        else:
            y = reward + 0.99 * np.max(self.sess.run(self.Q, feed_dict={self.state_ph:[new_state]})[0])

        if self.current_used == 1 or self.current_used == 3 and random.random() < self.L: # reward shaping q update
            teacher_action = self.teacher.egreedy_action(state, 1 - self.C)
            y += self.B * (self.R if teacher_action == action else -self.R)

        a = [0, 0]
        a[action] = 1
        self.replay.append((state, a, y))
        if len(self.replay) > REPLAY_SIZE:
            self.replay.popleft()
        if len(self.replay) > BATCH_SIZE:
            self.train()

    def train(self):
        with tf.variable_scope('MultiBandit'):
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