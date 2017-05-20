# import os
import gym
from gym import wrappers
from collections import deque
import random
import numpy as np

import tensorflow as tf

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''     
        batch_size specifies the number of experiences to add 
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least 
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                         tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradients = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradients)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.get_num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        net = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=400,
                activation_fn=tf.nn.relu)
        net = tf.contrib.layers.fully_connected(
                inputs=net,
                num_outputs=300,
                activation_fn=tf.nn.relu)
        # Final layer weights are init to uniform
        w_init = tf.Variable(tf.random_uniform( \
             [300, self.a_dim], minval=-0.003, maxval=0.003))
        out = tf.nn.tanh(tf.matmul(net, w_init))
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradients: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
         return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.squared_difference(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate
        ).minimize(self.loss)

        # Get the gradient of the nete w.r.t. the action.
        # For each action in the minibatch (i.e. for each x in xs)
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)
        net = tf.contrib.layers.fully_connected(
                inputs=inputs,
                num_outputs=400,
                activation_fn=tf.nn.relu
        )
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        weights_1 = tf.Variable(tf.random_normal( \
            [400, 300]))
        weights_2 = tf.Variable(tf.random_normal( \
            [self.a_dim, 300]))
        biases_2 = tf.Variable(tf.random_normal( \
            [300]))
        net = tf.nn.relu(tf.add(
            tf.matmul(net, weights_1),
            tf.nn.xw_plus_b(action, weights_2, biases_2)
        ))
        # Linear layer connected to 1 output representing Q(s, a)
        # Weights are init to Uniform [-3e-3, 3e-3]
        w_init = tf.Variable(tf.random_uniform( \
            [300, 1], minval=-0.003, maxval=0.003))
        out = tf.matmul(net, w_init)
        # out = tf.contrib.layers.fully_connected(
        #         inputs=net,
        #         num_outputs=1,
        #         weights_initializer=w_init)

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    summary_ops = tf.Summary()
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("DDPG/Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("DDPG/Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in xrange(MAX_EPISODES):
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in xrange(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, 3))) + (1./(1. + i))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch)
                )

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA*target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1))
                )
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]:ep_reward,
                    summary_vars[1]:ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j))

                break

def main(_):
    # if not os.path.exists(MONITOR_DIR):
    #     os.makedirs(MONITOR_DIR)

    # if not os.path.exists(SUMMARY_DIR):
    #     os.makedirs(SUMMARY_DIR)

    np.random.seed(RANDOM_SEED)
    env = gym.make(ENV_NAME)
    env.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    assert(env.action_space.high == -env.action_space.low)
    if GYM_MONITOR_EN:
        if not RENDER_ENV:
            env = wrappers.Monitor(
                env, MONITOR_DIR, video_callable=None, force=True
            )
        else:
            env = wrappers.Monitor(env, MONITOR_DIR, force=True)

    with tf.Session() as sess:
        env.reset() # circumvening TLS static error
        tf.set_random_seed(RANDOM_SEED)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars)

        train(sess, env, actor, critic)

    if GYM_MONITOR_EN:
        env.monitor.close()

if __name__ == '__main__':
    tf.app.run()