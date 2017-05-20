from __future__ import division

import numpy as np
import random
import matplotlib
matplotlib.use('Qt4agg')
import gym
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import os

# from gridworld import gameEnv
# execfile("gridworld.py")


# env = gameEnv(partial=False,size=5)
env = gym.make('Pong-v0')
# env = gym.envs.make("Breakout-v0")

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
            self.output = tf.image.resize_images(self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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

class Qnetwork():
    def __init__(self, h_size):
        # The network receives a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        # self.scalarInput = tf.placeholder(shape=[None, 7056], dtype=tf.float32)
        # self.scalarInput = tf.div(self.scalarInput, 255.0)
        # self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 1])
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4],
            padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(\
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
            padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(\
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
            padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1],
            padding='VALID', biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate
        # advantage and value stream
        self.streamAC, self.streamVC = tf.split(3, 2, self.conv4)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, env.action_space.n]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(
            self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the
        # target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.action_space.n, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size=500000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states, [21168])
    # return np.reshape(states, [7056])
    
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(
            tfVars[idx+total_vars//2].assign((var.value()*tau)\
            + ((1-tau)*tfVars[idx+total_vars//2].value())))

    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 1 # Starting chance of random action
endE = 0.1 # Final chance of random action
annealing_steps = 100000. # How many steps of training to reduce startE to endE.
num_episodes = 100000 # How many epsiodes of game environment to train network with.
pre_train_steps = 100000 # How many steps of random actions beofre training begins.
max_epLength = 300 # The max allowed length of our episode.
load_model = False # Wehther to load a saved model.
path = "./dqn_gym_smry" # The path to save our model to.
h_size = 512 # The size of the final convolutional layer before splitting it into
             # Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

sp = StateProcessor()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

summary_writer = tf.summary.FileWriter("train_dqn_gym_sumry")
# Set the rate of random action decrease.
eps = startE
stepDrop = (startE - endE)/annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
valueList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    updateTarget(targetOps, sess) # Set the target network to be equal to the primary network.

    for i_episode in xrange(num_episodes):
        episodeBuffer = experience_buffer()
        # Reset environment and get first new observation
        # s = env.reset()
        state = env.reset()
        state = cv2.resize(state, (84, 84))
        # state = sp.process(sess, state)
        state = processState(state)
        # observation = np.stack([observation_p] * 4, axis=2)
        # observations = np.array([observation] * 2)
        done = False
        rAll = 0
        j_stepEnv = 0
        # The Q-Network
        # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
        while j_stepEnv < max_epLength:
            j_stepEnv += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < eps or total_steps < pre_train_steps:
                action = env.action_space.sample()
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[state]})[0]
            next_state, reward, done, _ = env.step(action)
            next_state = cv2.resize(next_state, (84, 84))
            # next_state = sp.process(sess, next_state)
            # next_state = cv2.resize(next_state, (84, 84))
            next_state = processState(next_state)
            total_steps += 1
            # Save the experience to our episode buffer
            episodeBuffer.add(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))
            if total_steps > pre_train_steps:
                if eps > endE:
                    eps -= stepDrop

                if total_steps % (update_freq) == 0:
                    env.render()
                    # Get a random batch of experiences.
                    trainBatch = myBuffer.sample(batch_size)
                    # Below we perform the Double DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y*doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel,
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    updateTarget(targetOps, sess) # Set the target network to be equal to the primary network
                    rAll += reward
                    print "\rStep {} @ Episode {}/{} ({})".format(
                        j_stepEnv, i_episode + 1, num_episodes, rAll
                    )
                    state = next_state
                    valueList.append(doubleQ)

                    if done == True:
                        break

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j_stepEnv)
        rList.append(rAll)
        if i_episode % 10 == 0 and i_episode != 0:
            if i_episode % 1000 == 0:
                saver.save(sess,path+'/model-'+str(i_episode)+'.cptk')
                print("Saved Model")
            if len(rList) % 10 == 0:
                print "\rTotal steps{}, Averaged Reward List {} and eps {}".format(
                    total_steps, np.mean(rList[-10:]), eps)
            mean_reward = np.mean(rList[-10:])
            mean_length = np.mean(jList[-10:])
            if valueList == []:
                mean_value = 0
            else:
                mean_value = np.mean(valueList[-10:])
            summary = tf.Summary()
            summary.value.add(tag='Perf_pong/Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Perf_pong/Length', simple_value=float(mean_length))
            summary.value.add(tag='Perf_pong/Value', simple_value=float(mean_value))
            summary_writer.add_summary(summary, i_episode)
            summary_writer.flush()
    saver.save(sess,path+'/model-'+str(i_episode)+'.cptk')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)

# plt.plot(rMean)
# plt.show()