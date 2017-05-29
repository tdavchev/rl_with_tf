import os
import gym
from gym import wrappers
import numpy as np
# ========================================
#   Utility Parameters
# circumvening TLS static error
# Init here temporary until os update...
# ========================================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pong-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# Seed
RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
env = gym.make(ENV_NAME)
env.seed(RANDOM_SEED)

if GYM_MONITOR_EN:
    if not RENDER_ENV:
        env = wrappers.Monitor(
            env, MONITOR_DIR, video_callable=None, force=True
        )
    else:
        env = wrappers.Monitor(env, MONITOR_DIR, force=True)

    # env.monitor.close()


env.reset()
#env.render()
from collections import deque
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim

# ==========================
#   Training Parameters
# ==========================
# Update Frequency
update_freq = 4
# Max training steps
MAX_EPISODES = 8000
# Max episode length
MAX_EP_STEPS = 250000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00025
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.00025
# Contributes to the nitial random walk
MAX_START_ACTION_ATTEMPTS = 30
# Update params
FREEZE_INTERVAL = 10000
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Starting chance of random action
START_EPS = 1
# Final chance of random action
END_EPS = 0.05
# How many steps of training to reduce startE to endE.
ANNEALING = 1000000
# Number of options
OPTION_DIM = 8
# Pretrain steps
PRE_TRAIN_STEPS = 50000
# Size of replay buffer
BUFFER_SIZE = 1000000
# Minibatch size
MINIBATCH_SIZE = 32

class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    # D. Britz Implementation
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, (84, 84),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State
        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

class ReplayBuffer(object):
    """ Class, taken from https://github.com/jeanharb/option_critic
       A replay memory consisting of circular buffers for observed images,
       actions, and rewards.
    """
    def __init__(self, width, height, seed, max_steps=1000, phi_length=4):
        """Construct a DataSet.

        Arguments:
        width, height - image size
        max_steps - the number of time steps to store
        phi_length - number of images to concatenate into a state
        rng - initialized numpy random number generator, used to
        choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.phi_length = phi_length
        np.random.seed(RANDOM_SEED)

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((height, width, max_steps), dtype='uint8')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype='float32')
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.

        Arguments:
        img -- observed image
        action -- action chosen by the agent
        reward -- reward received after taking the action
        terminal -- boolean indicating whether the episode ended
        after this time step
        """
        self.imgs[:, :, self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1

        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=2, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype='float32')
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=2, mode='wrap')
        phi[-1] = img
        return phi

    def random_batch(self, batch_size, random_selection=False):
        """Return corresponding states, actions, rewards, terminal status, and
            next_states for batch_size randomly chosen state transitions.
        """
        # Allocate the response.
        states = np.zeros(
            (batch_size, self.height, self.width, self.phi_length), dtype='uint8')
        actions = np.zeros((batch_size), dtype='int32')
        rewards = np.zeros((batch_size), dtype='float32')
        terminal = np.zeros((batch_size), dtype='bool')
        next_states = np.zeros(
            (batch_size, self.height, self.width, self.phi_length), dtype='uint8')

        count = 0
        indices = np.zeros((batch_size), dtype='int32')

        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = np.random.randint(self.bottom, self.bottom + self.size - self.phi_length)

            initial_indices = np.arange(index, index + self.phi_length)
            transition_indices = initial_indices + 1
            end_index = index + self.phi_length - 1

            if np.any(self.terminal.take(initial_indices[0:-1], mode='wrap')):
                continue

            indices[count] = index

            # Add the state transition to the response.
            states[count] = self.imgs.take(initial_indices, axis=2, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            next_states[count] = self.imgs.take(transition_indices, axis=2, mode='wrap')
            count += 1

        return states, actions, rewards, next_states, terminal

class OptionsNetwork(object):
    def __init__(self, sess, h_size, temp, state_dim, action_dim,
                 option_dim, learning_rate, tau, entropy_reg=0.01, clip_delta=0):
        self.sess = sess
        self.h_size = h_size
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.o_dim = option_dim
        # self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.temp = temp

        # State Network
        self.inputs = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="inputs")
        scaledImage = tf.to_float(self.inputs) / 255.0
        self.next_inputs = tf.placeholder(
            shape=[None, 84, 84, 4],
            dtype=tf.uint8,
            name="next_inputs")
        next_scaledImage = tf.to_float(self.next_inputs) / 255.0
        with tf.variable_scope("state_out") as scope:
            self.state_out = self.apply_state_model(scaledImage)
            scope.reuse_variables()
            self.next_state_out = self.apply_state_model(next_scaledImage)

        with tf.variable_scope("q_out") as q_scope:
            # self.state_out = self.create_state_network(scaledImage)
            self.Q_out = self.apply_q_model(self.state_out)
            q_scope.reuse_variables()
            self.next_Q_out = self.apply_q_model(self.next_state_out)

        self.network_params = tf.trainable_variables()[:-2]
        self.Q_params = tf.trainable_variables()[-2:]

        # Prime Network
        self.target_inputs = tf.placeholder(
            shape=[None, 84, 84, 4],
            dtype=tf.uint8,
            name="target_inputs")
        target_scaledImage = tf.to_float(self.target_inputs) / 255.0
        self.target_state_out_holder = self.create_state_network(target_scaledImage)
        self.target_state_out_holder = tf.squeeze(self.target_state_out_holder, [1])
        with tf.variable_scope("target_q_out") as target_q_scope:
            self.target_Q_out = self.apply_target_q_model(self.target_state_out_holder)
            target_q_scope.reuse_variables()
        self.target_network_params = tf.trainable_variables()[
            len(self.network_params)+len(self.Q_params):-2]
        self.target_Q_params = tf.trainable_variables()[-2:]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.update_target_q_params = \
            [self.target_Q_params[i].assign(
                tf.multiply(self.Q_params[i], self.tau) + \
                tf.multiply(self.target_Q_params[i], 1. - self.tau))
             for i in range(len(self.target_Q_params))]

        # gather_nd should also do, though this is sexier
        self.option = tf.placeholder(tf.int32, [None, 1], name="option")
        self.action = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action")
        self.actions_onehot = tf.squeeze(tf.one_hot(self.action, self.a_dim, dtype=tf.float32), [1])
        self.options_onehot = tf.squeeze(tf.one_hot(self.option, self.o_dim, dtype=tf.float32), [1])

        # Action Network
        self.action_input = tf.concat(
            [self.state_out, self.state_out, self.state_out, self.state_out,
             self.state_out, self.state_out, self.state_out, self.state_out], 1)
        self.action_input = tf.reshape(self.action_input, shape=[-1, self.o_dim, 1, self.h_size])
        oh = tf.reshape(self.options_onehot, shape=[-1, self.o_dim, 1])
        self.action_input = tf.reshape(
            tf.reduce_sum(
                tf.squeeze(
                    self.action_input, [2]) * oh, [1]),
            shape=[-1, 1, self.h_size])

        self.action_probs = tf.contrib.layers.fully_connected(
            inputs=self.action_input,
            num_outputs=self.a_dim,
            activation_fn=tf.nn.softmax)
        self.action_probs = tf.squeeze(self.action_probs, [1])
        self.action_params = tf.trainable_variables()[
            len(self.network_params) + len(self.target_network_params) + \
            len(self.Q_params) + len(self.target_Q_params):]
        # always draws 0 ...
        # self.sampled_action = tf.argmax(tf.multinomial(self.action_probs, 1), axis=1)

        # Termination Network
        with tf.variable_scope("termination_probs") as term_scope:
            self.termination_probs = self.apply_termination_model(tf.stop_gradient(self.state_out))
            term_scope.reuse_variables()
            self.next_termination_probs =  self.apply_termination_model(tf.stop_gradient(self.next_state_out))

        self.termination_params = tf.trainable_variables()[-2:]

        self.option_term_prob = tf.reduce_sum(
            self.termination_probs * self.options_onehot, [1])
        self.next_option_term_prob = tf.reduce_sum(
            self.next_termination_probs * self.options_onehot, [1])

        self.reward = tf.placeholder(tf.float32, [None, 1], name="reward")
        self.done = tf.placeholder(tf.float32, [None, 1], name="done")
        # self.disc_option_term_prob = tf.placeholder(tf.float32, [None, 1])

        disc_option_term_prob = tf.stop_gradient(self.next_option_term_prob)

        y = tf.squeeze(self.reward, [1]) + \
            tf.squeeze((1 - self.done), [1]) * \
            GAMMA * (
                (1 - disc_option_term_prob) * \
                tf.reduce_sum( self.target_Q_out * self.options_onehot, [1] ) + \
                disc_option_term_prob * \
                tf.reduce_max( self.target_Q_out, reduction_indices=[1] ) )

        y = tf.stop_gradient(y)

        option_Q_out = tf.reduce_sum( self.Q_out * self.options_onehot, [1] )
        td_errors = y - option_Q_out
        # self.td_errors = tf.squared_difference(self.y, self.option_Q_out)

        if clip_delta > 0:
            quadratic_part = tf.minimum(abs(td_errors), clip_delta)
            linear_part = abs(td_errors) - quadratic_part
            td_cost = 0.5 * quadratic_part ** 2 + clip_delta * linear_part
        else:
            td_cost = 0.5 * td_errors ** 2

        # critic updates
        self.critic_cost = tf.reduce_sum(td_cost)
        critic_params = self.network_params + self.Q_params
        grads = tf.gradients(self.critic_cost, critic_params)
        self.critic_updates = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.95, epsilon=0.01).apply_gradients(zip(grads, critic_params))

        # actor updates
        self.value = tf.stop_gradient(tf.reduce_max(self.Q_out, reduction_indices=[1]))
        self.disc_q = tf.stop_gradient(tf.reduce_sum(self.Q_out * self.options_onehot, [1]))
        self.picked_action_prob = tf.reduce_sum(self.action_probs * self.actions_onehot, [1])
        actor_params = self.termination_params + self.action_params
        entropy = - tf.reduce_sum(self.action_probs * tf.log(self.action_probs))
        policy_gradient = - tf.reduce_sum(tf.log(self.picked_action_prob) * y) - \
            entropy_reg * entropy
        self.term_gradient = tf.reduce_sum(self.option_term_prob*(self.disc_q - self.value))
        self.loss = self.term_gradient+policy_gradient
        grads = tf.gradients(self.loss, actor_params)
        self.actor_updates = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.95, epsilon=0.01).apply_gradients(zip(grads, actor_params))

    def apply_state_model(self, input_image):
        with tf.variable_scope("input"):   
            output = self.state_model(
                input_image, [[8, 8, 4, 32], [4, 4, 32, 64], [3, 3, 64, 64], [7, 7, 64, 512]], [[512, 512]])
        return output

    def apply_q_model(self, input):
        with tf.variable_scope("q_input"):   
            output = self.q_model(input, [self.h_size, self.o_dim])
        return output

    def apply_target_q_model(self, input):
        with tf.variable_scope("target_q_input"):   
            output = self.target_q_model(input, [self.h_size, self.o_dim])
        return output

    def apply_termination_model(self, input):
        with tf.variable_scope("term_input"):
            output = self.termination_model(input, [self.h_size, self.o_dim])
        return output

    def state_model(self, input, kernel_shapes, weight_shapes):
        weights1 = tf.get_variable(
            "weights1", kernel_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        weights2 = tf.get_variable(
            "weights2", kernel_shapes[1],
            initializer=tf.contrib.layers.xavier_initializer())
        weights3 = tf.get_variable(
            "weights3", kernel_shapes[2],
            initializer=tf.contrib.layers.xavier_initializer())
        weights4 = tf.get_variable(
            "weights4", kernel_shapes[3],
            initializer=tf.contrib.layers.xavier_initializer())
        weights5 = tf.get_variable(
            "weights5", weight_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "q_bias1", weight_shapes[0][1],
            initializer=tf.constant_initializer())
        # Convolve
        conv1 = tf.nn.relu(tf.nn.conv2d(input, weights1, strides=[1, 4, 4, 1], padding='VALID'))
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights2, strides=[1, 2, 2, 1], padding='VALID'))
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights3, strides=[1, 1, 1, 1], padding='VALID'))
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights4, strides=[1, 1, 1, 1], padding='VALID'))
        net = tf.nn.relu(tf.nn.xw_plus_b(tf.squeeze(tf.squeeze(conv4, [1]), [1]), weights5, bias1))

        return net

    def target_q_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "target_q_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "target_q_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(tf.squeeze(input, [1]), weights1, bias1)

    def q_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "q_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "q_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(input, weights1, bias1)

    def termination_model(self, input, weight_shape):
        weights1 = tf.get_variable(
            "term_weights1", weight_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "term_bias1", weight_shape[1],
            initializer=tf.constant_initializer())
        return tf.nn.sigmoid(tf.nn.xw_plus_b(input, weights1, bias1))

    def create_state_network(self, scaledImage):
        # inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        conv1 = slim.conv2d( \
            inputs=scaledImage, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
            padding='VALID', biases_initializer=None)
        conv2 = slim.conv2d(\
            inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None)
        conv3 = slim.conv2d(\
            inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None)
        conv4 = slim.convolution2d( \
            inputs=conv3, num_outputs=self.h_size, kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None)
        net = tf.contrib.layers.fully_connected(
            inputs=conv4,
            num_outputs=self.h_size,
            activation_fn=tf.nn.relu)

        return net

    def predict(self, inputs):
        return self.sess.run(self.Q_out, feed_dict={
            self.inputs: [inputs]
        })

    def predict_action(self, inputs, option):
        return self.sess.run(self.action_probs, feed_dict={
            self.inputs: inputs,
            self.option: option
        })

    def predict_termination(self, inputs, option):
        return self.sess.run(
            [self.option_term_prob, self.Q_out],
            feed_dict={
                self.inputs: inputs,
                self.option: option
            })

    def train_actor(self, inputs, target_inputs, options, actions, r, done):
        return self.sess.run(self.actor_updates, feed_dict={
            self.inputs: inputs,
            self.next_inputs: target_inputs,
            self.target_inputs: target_inputs,
            self.option: options,
            self.action: actions,
            self.reward: r,
            self.done: done
        })

    def train_critic(self, inputs, target_inputs, options, r, done):
        return self.sess.run([self.critic_cost, self.critic_updates], feed_dict={
            self.inputs: inputs,
            self.next_inputs: target_inputs,
            self.target_inputs: target_inputs,
            self.reward: r,
            self.option: options,
            self.done: done
        })

    def update_target_network(self):
        self.sess.run([self.update_target_network_params, self.update_target_q_params])

# ===========================
#   Tensorflow Summary Opself.model
# ===========================
def build_summaries():
    summary_ops = tf.Summary()
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("DOCA/Qmax Value", episode_ave_max_q)
    episode_termination_ratio = tf.Variable(0.)
    tf.summary.scalar("DOCA/Term Ratio", episode_termination_ratio)
    tot_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Total Reward", tot_reward)
    cum_reward = tf.Variable(0.)
    tf.summary.scalar("DOCA/Cummulative Reward", tot_reward)
    rmng_frames = tf.Variable(0.)
    tf.summary.scalar("DOCA/Remaining Frames", rmng_frames)

    summary_vars = [
        episode_reward, episode_ave_max_q,
        episode_termination_ratio, tot_reward, cum_reward, rmng_frames]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def get_reward(reward):
    if reward < 0:
        score = -1
    elif reward > 0:
        score = 1
    else:
        score = 0

    return score, reward

# ===========================
#   Agent Training
# ===========================
def train(sess, env, option_critic):#, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
    np.random.seed(RANDOM_SEED)

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    option_critic.update_target_network()
    # critic.update_target_network()

    # State processor
    state_processor = StateProcessor()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(84, 84, RANDOM_SEED, BUFFER_SIZE, 4)
    # Set the rate of random action decrease.
    eps = START_EPS
    stepDrop = (START_EPS - END_EPS)/ANNEALING

    total_steps = 0
    print_option_stats = False

    action_counter = [{j:0 for j in range(env.action_space.n)} for i in range(OPTION_DIM)]
    total_reward = 0
    for i in xrange(MAX_EPISODES):
        s = env.reset() # note I'm using only one step, original uses 4
        s = state_processor.process(sess, s)
        s = np.stack([s] * 4, axis=2)
        current_option = 0
        current_action = 0
        new_option = np.argmax(option_critic.predict(s))
        #+ (1./(1. + i)) # state has more than 3 features in pong
        done = False
        termination = True
        ep_reward = 0
        ep_ave_max_q = 0
        termination_counter = 0
        since_last_term = 1

        for j in xrange(MAX_EP_STEPS):
            # if RENDER_ENV:
            #     env.render()

            if termination:
                if print_option_stats:
                    print "terminated -------", since_last_term,

                termination_counter += 1
                since_last_term = 1
                current_option = np.random.randint(OPTION_DIM) \
                    if np.random.rand() < eps else new_option
            else:
                if print_option_stats:
                    print "keep going"

                since_last_term += 1

            action_probs = option_critic.predict_action([s], np.reshape(current_option, [1, 1]))[0]
            current_action = np.argmax(np.random.multinomial(1, action_probs))
            if print_option_stats:
                print current_option
                if True:
                    action_counter[current_option][current_action] += 1
                    data_table = []
                    option_count = []
                    for ii, aa in enumerate(action_counter):
                        s3 = sum([aa[a] for a in aa])
                        if s3 < 1:
                            continue

                        print ii, aa, s3
                        option_count.append(s3)
                        print [str(float(aa[a])/s3)[:5] for a in aa]
                        data_table.append([float(aa[a])/s3 for a in aa])
                        print

                    print

            s2, reward, done, info = env.step(current_action)
            s2 = state_processor.process(sess, s2)
            s2 = np.append(s[:, :, 1:], np.expand_dims(s2, 2), axis=2)
            score, reward = get_reward(reward)
            total_steps += 1

            replay_buffer.add_sample(s[:, :, -1], current_option, score, done)

            term = option_critic.predict_termination([s2], [[current_option]])
            option_term_ps, Q = term[0], term[1]
            ep_ave_max_q += np.max(Q)
            new_option = np.argmax(Q)
            randomize = np.random.uniform(size=np.asarray([0]).shape)
            termination = option_term_ps if termination >= randomize else randomize
            if total_steps < PRE_TRAIN_STEPS:
                termination = 1

            if total_steps > PRE_TRAIN_STEPS:
                if eps > END_EPS:
                    eps -= stepDrop

                # done in the original paper, actor is trained on current data
                # critic trained on sampled one
                _ = option_critic.train_actor(
                    [s], [s2],
                    np.reshape(current_option, [1, 1]),
                    np.reshape(current_action, [1, 1]),
                    np.reshape(score, [1, 1]),
                    np.reshape(done+0, [1, 1]))

                if total_steps % (update_freq) == 0:
                    if RENDER_ENV:
                        env.render()

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if len(replay_buffer) > MINIBATCH_SIZE:
                        s_batch, o_batch, score_batch, s2_batch, done_batch = \
                            replay_buffer.random_batch(MINIBATCH_SIZE)

                        _ = option_critic.train_critic(
                            s_batch, s2_batch,
                            np.reshape(o_batch, [MINIBATCH_SIZE, 1]),
                            np.reshape(score_batch, [MINIBATCH_SIZE, 1]),
                            np.reshape(done_batch+0, [MINIBATCH_SIZE, 1]))

                if total_steps % (FREEZE_INTERVAL) == 0:
                    # Update target networks
                    option_critic.update_target_network()

            s = s2
            ep_reward += reward
            total_reward += reward

            if done:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]:ep_reward,
                    summary_vars[1]:ep_ave_max_q / float(j),
                    summary_vars[2]:float(termination_counter) / float(j),
                    summary_vars[3]:total_reward,
                    summary_vars[4]:total_reward/float(i+1),
                    summary_vars[5]:(MAX_EP_STEPS-j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                break

        print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
            '| Qmax: %.4f' % (ep_ave_max_q / float(j)), \
            ' | Cummulative Reward: %.1f' % (total_reward/(i+1)), \
            ' | %d Remaining Frames: ' %(MAX_EP_STEPS-j), \
            ' Epsilon: %.4f'%eps

def main(_):
    # if not os.path.exists(MONITOR_DIR):
    #     os.makedirs(MONITOR_DIR)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if state_dim == 210:
        # state_dim *= env.observation_space.shape[1] # for grey scale
        state_dim = 84 * 84 * 4
    # action_bound = env.action_space.high
    # Ensure action bound is symmetric
    # assert(env.action_space.high == -env.action_space.low)


    with tf.Session() as sess:
        tf.set_random_seed(RANDOM_SEED)
        # sess, h_size, temp, state_dim, action_dim, option_dim, action_bound, learning_rate, tau
        option_critic = OptionsNetwork(sess, 512, 1, state_dim, action_dim, 8, ACTOR_LEARNING_RATE, TAU, clip_delta=1)
        # critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars)

        train(sess, env, option_critic)#, critic)

    # if GYM_MONITOR_EN:
    #     env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
