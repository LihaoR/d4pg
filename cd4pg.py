#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:20:24 2018

@author: lihaoruo
"""
import threading
import numpy as np
import tensorflow as tf
import scipy.signal
import gym
from time import sleep

GLOBAL_STEP = 0

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class AC_Network():
    def __init__(self,s_size,a_size,scope,atrainer,ctrainer):
        with tf.variable_scope(scope):
            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.inputs_ = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.reward = tf.placeholder(shape=[None, 1],dtype=tf.float32)
            
            self.policy = self.build_a(self.inputs)
            self.value  = self.build_c(self.inputs, self.policy)
            #self.policy_ = self.build_a(self.inputs_, reuse=True, custom_getter=ema_getter)
            #self.value_  = self.build_c(self.inputs_, self.policy_, reuse=True, custom_getter=ema_getter)
            
            if scope != 'global':
                self.memory = np.zeros((MEMORY_CAPACITY, s_size * 2 + a_size + 1), dtype=np.float32)
                self.pointer = 0
                local_varsa = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/actor')
                local_varsc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/critic')
                target_update = [ema.apply(local_varsa), ema.apply(local_varsc)]
                
                self.policy_ = self.build_a(self.inputs_, reuse=True, custom_getter=ema_getter)
                self.value_  = self.build_c(self.inputs_, self.policy_, reuse=True, custom_getter=ema_getter)
                # actor 
                self.policy_loss = -tf.reduce_mean(self.value)
                self.gradientsa = tf.gradients(self.policy_loss,local_varsa)
                self.var_normsa = tf.global_norm(local_varsa)
                gradsa,self.grad_normsa = tf.clip_by_global_norm(self.gradientsa,40.0)
                global_varsa = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/actor')
                self.apply_gradsa = atrainer.apply_gradients(zip(gradsa, global_varsa))

                # critic
                with tf.control_dependencies(target_update):
                    self.target_v = self.reward + gamma * self.value_
                    self.value_loss = tf.reduce_sum(tf.square(self.target_v - self.value))
                    self.gradientsc = tf.gradients(self.value_loss,local_varsc)
                    self.var_normsc = tf.global_norm(local_varsc)
                    gradsc,self.grad_normsc = tf.clip_by_global_norm(self.gradientsc,40.0)
                    global_varsc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global/critic')
                    self.apply_gradsc = ctrainer.apply_gradients(zip(gradsc, global_varsc))
                    
    
    def build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1a', trainable=trainable)
            net = tf.layers.dense(net, 30, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net, a_size, activation=tf.nn.tanh, name='a1', trainable=trainable)
            return tf.multiply(a, a_bound, name='scaled_a')

    def build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1c',trainable=trainable)
            w1_s = tf.get_variable('w1_s', [30, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [a_size, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(net, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

class Worker():
    def __init__(self,env,name,s_size,a_size,atrainer,ctrainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.atrainer = atrainer
        self.ctrainer = ctrainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.local_AC = AC_Network(s_size,a_size,self.name,atrainer,ctrainer)
        self.update_local_opsa = update_target_graph('global/actor', self.name+'/actor')
        self.update_local_opsc = update_target_graph('global/critic',self.name+'/critic')
        self.env = env
        
    def train(self,indices,sess,gamma):
        bt  = self.local_AC.memory[indices, :]
        bs  = bt[:, :s_size]
        ba  = bt[:, s_size:s_size+a_size]
        br  = bt[:, -s_size-1:-s_size]
        bs_ = bt[:, -s_size:]

        p_l,_ = sess.run([self.local_AC.policy_loss,self.local_AC.apply_gradsa],
                         feed_dict={self.local_AC.inputs:bs})
        v_l,_ = sess.run([self.local_AC.value_loss, self.local_AC.apply_gradsc], 
                         feed_dict={self.local_AC.inputs:bs,
                                    self.local_AC.reward:br,
                                    self.local_AC.policy:ba,
                                    self.local_AC.inputs_:bs_})
        return v_l, p_l
        
    def work(self,gamma,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        var = 3
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run([self.update_local_opsa,self.update_local_opsc])
                episode_reward = 0
                episode_step_count = 0
                d = False
                s = self.env.reset()
                while not d:
                    #if self.name == "worker_1":
                    #    self.env.render()
                    GLOBAL_STEP += 1
                    a_dist = sess.run([self.local_AC.policy], feed_dict={self.local_AC.inputs:[s]})[0][0]
                    a = np.clip(np.random.normal(a_dist, var), -2, 2)
                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = s1
                    else:
                        s1 = s

                    self.local_AC.store_transition(s, a, r, s1)
                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    if total_steps % 10 == 0 and d != True and self.local_AC.pointer > MEMORY_CAPACITY:
                        var *= 0.9995
                        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
                        v_l, p_l = self.train(indices,sess,gamma)
                        sess.run([self.update_local_opsa,self.update_local_opsc])
                    if d == True:
                        print 'name', self.name, 'step', episode_count, 'reward', episode_reward, 'var', var
                        break

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

gamma = .99 
load_model = False
model_path = './a3model'

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]
a_bound = env.action_space.high
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
tf.reset_default_graph()

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

atrainer = tf.train.AdamOptimizer(learning_rate=0.0001)
ctrainer = tf.train.AdamOptimizer(learning_rate=0.001)

master_network = AC_Network(s_size,a_size,'global',None,None)
num_workers = 4
workers = []

for i in range(num_workers):
    env = gym.make(ENV_NAME)
    #env = env.unwrapped
    workers.append(Worker(env,i,s_size,a_size,atrainer,ctrainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    