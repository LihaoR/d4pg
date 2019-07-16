#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:12:24 2018

@author: lihaoruo
"""
import roboschool
from OpenGL import GLU
from time import sleep
import tensorflow as tf
import numpy as np
import gym
import threading
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
LR_A = 0.0001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 3000
BATCH_SIZE = 64
GLOBAL_NET_SCOPE = 'Global_Net'
ENV_NAME = 'RoboschoolInvertedPendulum-v1'
N_WORKERS = 2
N = 1

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, scope, globalD=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
                self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
                self.s_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
                self.quantile = 1.0 / N
                
                with tf.variable_scope('Actor'):
                    l = self._build_a(self.s, scope='eval', trainable=False)
                with tf.variable_scope('Critic'):
                    v = self._build_c(self.s, l, scope='eval', trainable=False)
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/eval')
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/eval')
        else:
            with tf.variable_scope(scope):
                self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
                self.s  = tf.placeholder(tf.float32, [None, s_dim], 's')
                self.s_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
                self.r  = tf.placeholder(tf.float32, [None, 1], 'r')
                self.quantile = 1.0 / N

                self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
                self.pointer = 0

                with tf.variable_scope('Actor'):
                    self.a = self._build_a(self.s, scope='eval', trainable=True)
                    self.a_ = self._build_a(self.s_, scope='target', trainable=False)
                with tf.variable_scope('Critic'):
                    self.q = self._build_c(self.s, self.a, scope='eval', trainable=True)
                    self.q_ = self._build_c(self.s_, self.a_, scope='target', trainable=False)
                self.Q = tf.reduce_sum(self.quantile * self.q, axis=1)

                # networks parameters
                self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Actor/eval')
                self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Actor/target')
                self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/Critic/eval')
                self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + '/Critic/target')
                # target net replacement
                self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                                     for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
                
                self.q_target = self.r + GAMMA * tf.stop_gradient(self.q_)
                td_error = tf.reduce_mean(tf.reduce_sum(tf.square(self.q - self.q_target),axis=1))
                self.c_grads = tf.gradients(td_error, self.ce_params)
                optc = tf.train.AdamOptimizer(LR_C, name='adamc')
                self.update_c_op = optc.apply_gradients(list(zip(self.c_grads, globalD.ce_params)))
                self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ce_params, globalD.ce_params)]

                a_loss = - tf.reduce_mean(self.Q)
                self.a_grads = tf.gradients(a_loss, self.ae_params)
                opta = tf.train.AdamOptimizer(LR_A, name='adama')
                self.update_a_op = opta.apply_gradients(list(zip(self.a_grads, globalD.ae_params)))
                self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.ae_params, globalD.ae_params)]


    def choose_action(self, s):
        return sess.run(self.a, {self.s: s[np.newaxis, :]})[0]

    def learn(self):
        sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        
        sess.run([self.update_a_op], feed_dict={self.s: bs, self.a: ba})
        sess.run([self.update_c_op], feed_dict={self.s: bs, self.a: ba, self.r:br, self.s_:bs_})
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        w_init = tf.random_normal_initializer(-1., .1)
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 200, activation=tf.nn.relu, kernel_initializer=w_init, name='l1', trainable=trainable)
            #net = tf.layers.dense(net, 100, activation=tf.nn.relu, kernel_initializer=w_init, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a1', trainable=trainable)
            return a# tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        w_init = tf.random_normal_initializer(-1., .1)
        with tf.variable_scope(scope):
            """
            n_l1 = 100
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            """
            inputs = tf.concat([s, a], axis=1)
            net = tf.layers.dense(inputs, 200, activation=tf.nn.relu, kernel_initializer=w_init, name='c1', trainable=trainable)
            #net = tf.layers.dense(net, 100, activation=tf.nn.relu, kernel_initializer=w_init, name='c2', trainable=trainable)
            return tf.layers.dense(net, N, trainable=trainable)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_low = env.action_space.low
print (a_bound, a_low)
var = 0.3  # control exploration

class Worker(object):
    def __init__(self, name, globalD):
        self.env = gym.make(ENV_NAME).unwrapped
        self.name = name
        self.D = DDPG(a_dim, s_dim, a_bound, name, globalD)
        self.var = var
        self.if_render = False
        
    def work(self):
        total_step = 0
        reward_list = []
        max_reward = 0
        while True:
            s = env.reset()
            ep_reward = 0
            done = False
            
            while not done:
                if self.if_render == True and self.name == '0':
                    env.render()
                a = self.D.choose_action(s)
                a = np.clip(np.random.normal(a, self.var), -1., 1.)    # add randomness to action selection for exploration
                s_, r, done, info = env.step(a)
                
                self.D.store_transition(s, a, r, s_)
            
                if self.D.pointer > MEMORY_CAPACITY:
                    #self.if_render = True
                    self.D.learn()
                
                total_step += 1
                s = s_
                ep_reward += r
                if done == True:
                    break
                
            reward_list.append(ep_reward)
            max_reward = max(np.max(reward_list), max_reward)
            if len(reward_list) > 100:
                reward_list.pop(0)
            if self.D.pointer > MEMORY_CAPACITY:
                if self.var > 0.1:
                    self.var *= 0.9999
                elif self.var > 0.01 and self.var <= 0.1:
                    self.var *= 0.999999
                else:
                    pass
            if total_step % 50 == 0:
                print('name              :', self.name)
                print('episode           :', total_step)
                print('explore           : %.2f' % self.var)
                print('mean reward       : %.2f' % np.mean(reward_list))
                print('max reward        :', np.max(reward_list))
                print('global max reward :', max_reward)
                print('---------------------------------------')

if __name__ == "__main__":
    sess = tf.Session()
    with tf.device("/gpu:0"):
        GLOBAL_ddpg = DDPG(a_dim, s_dim, a_bound, GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = str(i)   # worker name
            workers.append(Worker(i_name, GLOBAL_ddpg))

    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    COORD.join(worker_threads)

