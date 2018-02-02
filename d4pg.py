"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""


import tensorflow as tf
import numpy as np
import gym
import multiprocessing
import threading
import os
import shutil
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
GLOBAL_NET_SCOPE = 'Global_Net'
ENV_NAME = 'Pendulum-v0'
N_WORKERS = 2#multiprocessing.cpu_count()
GLOBAL_RUNNING_R = []
LOG_DIR = './log'
OUTPUT_GRAPH = True

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, scope, globalD=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
                self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
                self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
                with tf.variable_scope('Actor'):
                    self.a = self._build_a(self.S, scope='eval', trainable=False)
                with tf.variable_scope('Critic'):
                    q = self._build_c(self.S, self.a, scope='eval', trainable=False)
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/eval')
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/eval')
            #sess.run(tf.global_variables_initializer())
        else:
            with tf.variable_scope(scope):
                self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
                self.pointer = 0
                self.a_replace_counter, self.c_replace_counter = 0, 0

                self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
                self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
                self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
                self.R = tf.placeholder(tf.float32, [None, 1], 'r')

                with tf.variable_scope('Actor'):
                    self.a = self._build_a(self.S, scope='eval', trainable=True)
                    a_ = self._build_a(self.S_, scope='target', trainable=False)
                with tf.variable_scope('Critic'):
                    # assign self.a = a in memory when calculating q for td_error,
                    # otherwise the self.a is from Actor when updating Actor
                    q = self._build_c(self.S, self.a, scope='eval', trainable=True)
                    q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

                # networks parameters
                self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Actor/eval')
                self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/target')
                self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Critic/eval')
                self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/target')
                #print('aaaaaaaaeeeeeeeeeeeeee', self.ae_params)
                #print('cccccccceeeeeeeeeeeeee', self.ce_params)
                # target net replacement
                self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                     for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
            
                q_target = self.R + GAMMA * q_
                # in the feed_dic for the td_error, the self.a should change to actions in memory
                td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                #self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
                self.c_grads = tf.gradients(td_error, self.ce_params)
                optc = tf.train.AdamOptimizer(LR_C, name='adamc')
                self.update_c_op = optc.apply_gradients(list(zip(self.c_grads, globalD.ce_params)))
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ce_params, globalD.ce_params)]

                a_loss = - tf.reduce_mean(q)    # maximize the q
                #self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
                self.a_grads = tf.gradients(a_loss, self.ae_params)
                opta = tf.train.AdamOptimizer(LR_A, name='adama')
                self.update_a_op = opta.apply_gradients(list(zip(self.a_grads, globalD.ae_params)))
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ae_params, globalD.ae_params)]
                #sess.run(tf.global_variables_initializer())
                
            
    def choose_action(self, s):
        return sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        #self.sess.run(self.atrain, {self.S: bs})
        #self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        sess.run([self.update_a_op], feed_dict={self.S:bs})
        sess.run([self.update_c_op], feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
    
    #def pull_global(self, feed_dict):
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a1', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

#ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration

class Worker(object):
    def __init__(self, name, globalD):
        self.env = gym.make(ENV_NAME).unwrapped
        self.name = name
        self.D = DDPG(a_dim, s_dim, a_bound, name, globalD)
        
    def work(self):
        var = 3
        for i in range(MAX_EPISODES):
            s = env.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):
                if self.name == 'W_0':
                    env.render()

                # Add exploration noise
                a = self.D.choose_action(s)
                a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
                s_, r, done, info = env.step(a)
                
                self.D.store_transition(s, a, r/10, s_)
            
                if self.D.pointer > MEMORY_CAPACITY:
                    var *= .9995    # decay the action randomness
                    self.D.learn()
                    #self.D.pull_global()
            
                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS-1:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                    break

if __name__ == "__main__":
    sess = tf.Session()
    #sess.run(tf.initialize_all_variables())
    #sess.run(tf.global_variables_initializer())
    with tf.device("/cpu:0"):
        GLOBAL_ddpg = DDPG(a_dim, s_dim, a_bound, GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_ddpg))

    COORD = tf.train.Coordinator()
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
    
    