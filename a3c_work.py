"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow r1.3
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
# import gym
import os
import shutil
import matplotlib.pyplot as plt

import ball_on_plate_env as ball

from keras.layers import LSTM
import keras
from keras import backend as K

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.95
ENTROPY_BETA = 0.01
LR_A = 1e-5    # learning rate for actor
LR_C = 1e-5    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# env = gym.make(GAME)
env = ball.BallOnPlateEnv()

N_S = env.observation_space_shape[0]
N_A = env.action_space_shape[0]
A_BOUND = [env.action_space_low, env.action_space_high]


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.state = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
                    
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.state, 50, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a1 = tf.layers.dense(l_a, 50, tf.nn.relu6, kernel_initializer=w_init, name='la1')
            # l_a2 = LSTM(50, kernel_initializer=w_init, name='la2')(l_a1)
            l_a3 = tf.layers.dense(l_a1, 50, tf.nn.relu6, kernel_initializer=w_init, name='la3')
            mu = tf.layers.dense(l_a3, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a3, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.state, 20, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c1 = tf.layers.dense(l_c, 20, tf.nn.relu6, kernel_initializer=w_init, name='lc1')
            l_c2 = tf.layers.dense(l_c1, 20, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c2, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, state):  # run by a local
        state = state[np.newaxis, :]
        return SESS.run(self.A, {self.state: state})[0]


class Worker(object):
    def __init__(self, name, globalAC):

        self.name = name
        self.AC = ACNet(name, globalAC)
        self.gAC = globalAC

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []

        if self.name == 'W_0':
            self.envir = ball.BallOnPlate(showGUI=True, randomInitial=True)
        else:
            self.envir = ball.BallOnPlate(showGUI=False, randomInitial=True)

        ref_point = np.array([0., 0.])


        while not COORD.should_stop(): # and GLOBAL_EP < MAX_GLOBAL_EP:
            posOnPlate = self.envir.reset()
            err = ref_point - posOnPlate
            state = np.array([posOnPlate[0], posOnPlate[1], err[0] / 2., err[1] / 2., 0, 0, 0, 0])

            ep_r = 0.

            # for ep_t in range(MAX_EP_STEP):
            while True:

                a = self.AC.choose_action(state)
                posOnPlate, done = self.envir.step(a)

                err = ref_point - posOnPlate
                r = 4 - (err[0]**2 + err[1]**2 + (posOnPlate[0]-state[0])**2/self.envir.dt + (posOnPlate[1]-state[1])**2/self.envir.dt) / 100.
                # r = float(r) / 4.

                # print(r)
                new_state = np.array([posOnPlate[0], posOnPlate[1], err[0] / 2., err[1] / 2., 
                                      posOnPlate[0]-state[0], posOnPlate[1]-state[1], a[0], a[1]])

                if done:
                    pass
                    # r -= self.envir.time / self.envir.dt * 4
                else:
                    done = True if self.envir.time > 20 else False

                ep_r += r
                buffer_s.append(state)
                buffer_a.append(a)
                buffer_r.append(r)    # normalize
                # buffer_r.append((r+8)/8)    # normalize

                # print(total_step, self.envir.time)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.state: new_state[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.state: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                state = new_state
                total_step += 1
                if done:
                    # if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                    GLOBAL_RUNNING_R.append(ep_r)
                    # else:
                        # GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1

                    if GLOBAL_EP % 500 == 0:
                        save_path = saver.save(SESS, "./a3c.chkp")
                        print("Model saved in path: %s" % save_path)

                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

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

