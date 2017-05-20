import numpy as np
import matplotlib
matplotlib.use('Qt4agg')
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('CartPole-v0')
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class actor():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)

        hidden = slim.fully_connected(
            self.state_in,
            h_size,
            biases_initializer=None,
            activation_fn=tf.nn.relu)
        # output holds probabilities for all rewards
        self.output = slim.fully_connected(
            hidden,
            a_size,
            activation_fn=tf.nn.softmax,
            biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the rewards and chosen actions into the network
        #to compute the loss, and use it to update the network.
        # reward holder contains the d * q
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.batch_action_indexes = tf.range(0,tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(
            tf.reshape(self.output, [-1]),
            self.batch_action_indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.

myActor = actor(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the actor.

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []
    rendering = False
    best_avg_reward = 0
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            if running_reward / update_frequency > best_avg_reward:
                best_avg_reward = running_reward / update_frequency
            # print "\raverage reward per episode: {}".format(running_reward / update_frequency)
            if running_reward / update_frequency > 100 or rendering == True:
                env.render()
                rendering = True

            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myActor.output,feed_dict={myActor.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)for t in reversed(xrange(0, r.size)):

            s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myActor.reward_holder:ep_history[:,2],
                        myActor.action_holder:ep_history[:,1],myActor.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myActor.gradients, feed_dict=feed_dict)
                # collect a buffer of gradients
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    # update gradients
                    feed_dict= dictionary = dict(zip(myActor.gradient_holders, gradBuffer))
                    _ = sess.run(myActor.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_lenght.append(j)
                break

        
            #Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
    print "\r best avg reward: {}".format(best_avg_reward)