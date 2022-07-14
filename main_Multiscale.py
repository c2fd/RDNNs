# Copyright @Jia Zhao 2020. All right reserved.
# Distribution or disclosure without authorization is prohibited.
# Email: jia.zhao@usu.edu

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.integrate import odeint
from pyDOE import lhs
import argparse
parser = argparse.ArgumentParser('Train or Test Arg! ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.random.seed(1234)
tf.set_random_seed(1234567)
import random
random.seed(12345)
np.random.seed(12345)

import os.path
file_base_name = os.path.basename(__file__)
file_base_name_without_extension = os.path.splitext(file_base_name)[0]
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from pathlib import Path
checkpoints_file = str(Path.home()) + '/Dropbox/CodeFolderTensorFlow/Ckpts/Learn-ODECkpt/' + file_base_name_without_extension + '.ckpt'


def generate_data_Cubic(N_f, constant_dt):
    def real_f(y, t):
        dydt = [-0.1 * y[0] ** 3 + 2.0 * y[1] ** 3,
                -2.0 * y[0] ** 3 - 0.1 * y[1] ** 3]
        return dydt

    def visualize(sol_t, u_real, u_pred):
        dim = u_real.shape[1]
        print('dimension', dim)

        for i in range(dim):
            plt.subplot(3, dim, i + 1)
            plt.plot(sol_t, u_real[:, i:i + 1], label='real')
            plt.plot(sol_t, u_pred[:, i:i + 1], '--', label='predicted')
            plt.legend(loc=1, fontsize = 'xx-small')
            #plt.subplot(3, dim, 2 * dim + i + 1)
            #plt.plot(sol_t, u_real[:, i:i + 1] - u_pred[:, i:i + 1]) #plot error

        if not args.viz:
            plt.savefig('./figures/' + file_base_name_without_extension + timestr + '.png')
        else:
            plt.show()

    Test = []
    # N_f = 1000  # 0
    # constant_dt = 0.01
    from pyDOE import lhs
    lb = np.array([-2.5, -2.5])
    ub = np.array([2.5, 2.5])
    X_f = lb + (ub - lb) * lhs(2, N_f)

    for i in range(N_f):
        test_y0 = X_f[i]
        test_t = np.linspace(0, constant_dt, 100)
        test_sol = odeint(real_f, test_y0, test_t)

        test_data = np.hstack([test_sol[0, :], test_sol[-1, :]])
        if i == 0:
            Test = test_data
        else:
            Test = np.vstack([Test, test_data])

    test_dt = Test[:, 0:1] * 0 + constant_dt  # Test[:,0:1]
    test_u0 = Test[:, 0:2]
    test_u1 = Test[:, 2:4]

    print('data shape', test_u0.shape, test_u1.shape, test_dt.shape, Test.shape)
    train_data_file = "ode_train_data_Cubic_constant_" + str(constant_dt) + ".mat"
    scipy.io.savemat(train_data_file, {'dt': test_dt, 'u0': test_u0, 'u1': test_u1})

    NN = 1000
    test_t = np.linspace(0, 25, NN + 1)
    test_u0 = [2.0, 0.]

    return train_data_file, real_f, test_u0, test_t, visualize


def generate_data_Multiscale(N_f, constant_dt):

    def real_f(y, t):
        x1 = y[0]
        x2 = y[1]
        x3 = y[2]
        x4 = y[3]
        vareps = 0.1

        dydt = [-x2 - x3,
                x1 + 1 / 5 * x2,
                1 / 5 + x4 - 5 * x3,
                -x4 / vareps + x1 * x3 / vareps]
        return dydt

    def visualize(sol_t, u_real, u_pred):
        dim = u_real.shape[1]
        # print('dimension', dim)

        for i in range(dim):
            plt.plot(sol_t, u_real[:, i:i + 1], label='real', linewidth=2.5)
            plt.plot(sol_t, u_pred[:, i:i + 1],'--', label='predicted', linewidth=2.5)
            plt.xticks(fontsize = 'xx-large')
            plt.yticks(fontsize = 'xx-large')
            #plt.subplot(3, dim, 2 * dim + i + 1)
            #plt.plot(sol_t, u_real[:, i:i + 1] - u_pred[:, i:i + 1])

            if not args.viz:
                plt.savefig('./figures/' + file_base_name_without_extension + time.strftime("%Y%m%d-%H%M%S") + '_' + 'multiscale' + str(i) + '.png')
            else:
                plt.show()
            plt.clf()


    Test = []
    # N_f = 1000
    # constant_dt = 0.1

    lb = np.array([-15, -15, -5, -30])
    ub = np.array([ 15,  10, 25, 140])
    X_f = lb + (ub - lb) * lhs(len(lb), N_f)

    for i in range(N_f):
        test_y0 = X_f[i]
        test_t = np.linspace(0, constant_dt, 100)
        test_sol = odeint(real_f, test_y0, test_t)

        test_data = np.hstack([test_sol[0, :], test_sol[-1, :]])
        if i == 0:
            Test = test_data
        else:
            Test = np.vstack([Test, test_data])

    test_dt = Test[:, 0:1] * 0 + constant_dt  # Test[:,0:1]
    test_u0 = Test[:, 0:len(lb)]
    test_u1 = Test[:, len(lb):2 * len(lb)]

    print('data shape', test_u0.shape, test_u1.shape, test_dt.shape, Test.shape)
    train_data_file ="ode_train_data_multiscale_constant_" + str(constant_dt) + ".mat"
    scipy.io.savemat(train_data_file, {'dt': test_dt, 'u0': test_u0, 'u1': test_u1})

    NN = 5000
    test_t = np.linspace(0, 20, NN + 1)
    test_u0 = [2.4350451, 3.416925, -2.16129375, 3.4650658]
    return train_data_file, real_f, test_u0, test_t, visualize


def generate_data_Glycolytic(N_f, constant_dt):

    def real_f(x, t):  # x is 3 x 1
        J0 = 2.5
        k1 = 100.0
        k2 = 6.0
        k3 = 16.0
        k4 = 100.0
        k5 = 1.28
        k6 = 12.0
        k = 1.8
        kappa = 13.0
        q = 4
        K1 = 0.52
        psi = 0.1
        N = 1.0
        A = 4.0

        f1 = J0 - (k1 * x[0] * x[5]) / (1 + (x[5] / K1) ** q)
        f2 = 2 * (k1 * x[0] * x[5]) / (1 + (x[5] / K1) ** q) - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
        f3 = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
        f4 = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
        f5 = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4]
        f6 = -2 * (k1 * x[0] * x[5]) / (1 + (x[5] / K1) ** q) + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
        f7 = psi * kappa * (x[3] - x[6]) - k * x[6]

        f = np.array([f1, f2, f3, f4, f5, f6, f7])
        return f

    def visualize(sol_t, u_real, u_pred):
        dim = u_real.shape[1]
        print('dimension', dim)

        for i in range(dim):
            plt.plot(sol_t, u_real[:, i:i + 1], label='real')
            plt.plot(sol_t, u_pred[:, i:i + 1],'--', label='predicted')
            plt.legend(loc=1, fontsize = 'xx-small')
            #plt.subplot(3, dim, 2 * dim + i + 1)
            #plt.plot(sol_t, u_real[:, i:i + 1] - u_pred[:, i:i + 1])

            if not args.viz:
                plt.savefig('./figures/' + file_base_name_without_extension + timestr + '_' + str(i) + '.png')
            else:
                plt.show()
            plt.clf()


    Test = []
    # N_f = 1000
    # constant_dt = 0.1

    lb = np.array([0, 0, 0, 0, 0, 0.14, 0.05])
    ub = np.array([2, 3, 0.5, 0.5, 0.5, 2.67, 0.15])
    X_f = lb + (ub - lb) * lhs(len(lb), N_f)

    for i in range(N_f):
        test_y0 = X_f[i]
        test_t = np.linspace(0, constant_dt, 100)
        test_sol = odeint(real_f, test_y0, test_t)

        test_data = np.hstack([test_sol[0, :], test_sol[-1, :]])
        if i == 0:
            Test = test_data
        else:
            Test = np.vstack([Test, test_data])

    test_dt = Test[:, 0:1] * 0 + constant_dt  # Test[:,0:1]
    test_u0 = Test[:, 0:len(lb)]
    test_u1 = Test[:, len(lb):2 * len(lb)]

    print('data shape', test_u0.shape, test_u1.shape, test_dt.shape, Test.shape)
    train_data_file ="ode_train_data_case6_moredata_constant_" + str(constant_dt) + ".mat"
    scipy.io.savemat(train_data_file, {'dt': test_dt, 'u0': test_u0, 'u1': test_u1})

    NN = 1000
    test_t = np.linspace(0, 5, NN + 1)
    test_u0 = [1.1, 1.0, 0.075, 0.175, 0.25, 0.9, 0.095]

    return train_data_file, real_f, test_u0, test_t, visualize


def generate_data_Lorenz(N_f, constant_dt):

    def real_f(y, t):
        dydt = [10 * (y[1] - y[0]),
                y[0] * (28 - y[2]) - y[1],
                y[0] * y[1] - 8 / 3 * y[2]
                ]
        return dydt

    def visualize(sol_t, u_real, u_pred):
        dim = u_real.shape[1]
        print('dimension', dim)

        for i in range(dim):
            plt.plot(sol_t, u_real[:, i:i + 1], label='real')
            plt.plot(sol_t, u_pred[:, i:i + 1], '--', label='predicted')
            plt.legend(loc=1, fontsize='xx-small')
            # plt.subplot(3, dim, 2 * dim + i + 1)
            # plt.plot(sol_t, u_real[:, i:i + 1] - u_pred[:, i:i + 1])

            if not args.viz:
                plt.savefig('./figures/' + file_base_name_without_extension + timestr + '_' + str(i) + '.png')
            else:
                plt.show()
            plt.clf()

    Test = []
    # N_f = 10000
    # constant_dt = 0.05
    lb = np.array([-25.0, -25.0, 0])
    ub = np.array([+25.0, +25.0, 50])  # np.pi/2])
    X_f = lb + (ub - lb) * lhs(3, N_f)

    for i in range(N_f):
        test_y0 = X_f[i]
        test_t = np.linspace(0, constant_dt, 100)
        test_sol = odeint(real_f, test_y0, test_t)

        test_data = np.hstack([test_sol[0, :], test_sol[-1, :]])
        if i == 0:
            Test = test_data
        else:
            Test = np.vstack([Test, test_data])

    test_dt = Test[:, 0:1] * 0 + constant_dt  # Test[:,0:1]
    test_u0 = Test[:, 0:3]
    test_u1 = Test[:, 3:6]

    print('data shape', test_u0.shape, test_u1.shape, test_dt.shape, Test.shape)
    train_data_file = "ode_train_data_Lorenz_constant_" + str(constant_dt) + ".mat"
    scipy.io.savemat(train_data_file, {'dt': test_dt, 'u0': test_u0, 'u1': test_u1})

    NN = 1000
    test_t = np.linspace(0, 5, NN + 1)
    test_u0 = [-8, 7, 27]
    return train_data_file, real_f, test_u0, test_t, visualize


def generate_data_Hopf(N_f, constant_dt):
    def real_f(z, t):
        mu = z[0]
        x = z[1]
        y = z[2]
        dydt = [0, mu * x + y - x * (x ** 2 + y ** 2), -x + mu * y - y * (x ** 2 + y ** 2)]
        return dydt

    def visualize(sol_t, u_real, u_pred):
        dim = u_real.shape[2]
        print('dimension', dim)
        print(u_real.shape)
        print(u_pred.shape)

        for i in range(dim):
            plt.subplot(3, dim, i + 1)
            plt.plot(sol_t, u_real[:, i:i + 1])
            plt.subplot(3, dim, dim + i + 1)
            plt.plot(sol_t, u_pred[:, i:i + 1])
            plt.subplot(3, dim, 2 * dim + i + 1)
            plt.plot(sol_t, u_real[:, i:i + 1] - u_pred[:, i:i + 1])

            if not args.viz:
                plt.savefig('./figures/' + "hopf" + timestr + '_' + str(i) + '.png')
            else:
                plt.show()
            plt.clf()

    Test = []
    # N_f = 10000
    # constant_dt = 0.5  # 01

    lb = np.array([-1, -2., -1.])
    ub = np.array([1., 2., 1.])
    X_f = lb + (ub - lb) * lhs(3, N_f)

    for i in range(N_f):
        test_y0 = X_f[i]
        test_t = np.linspace(0, constant_dt, 100)
        test_sol = odeint(real_f, test_y0, test_t)

        test_data = np.hstack([test_sol[0, :], test_sol[-1, :]])
        if i == 0:
            Test = test_data
        else:
            Test = np.vstack([Test, test_data])

    test_dt = Test[:, 0:1] * 0 + constant_dt  # Test[:,0:1]
    test_u0 = Test[:, 0:3]
    test_u1 = Test[:, 3:6]

    print('data shape', test_u0.shape, test_u1.shape, test_dt.shape, Test.shape)
    train_data_file = "ode_train_data_Hopf_constant_" + str(constant_dt) + ".mat"
    scipy.io.savemat(train_data_file, {'dt': test_dt, 'u0': test_u0, 'u1': test_u1})

    # time points
    test_t = np.arange(0, 75, 0.1)
    # test_dt = t_star[1] - t_star[0]
    # initial condition
    test_u0 = np.array([[-0.15, 2, 0],
                           [-0.05, 2, 0],

                           [.05, .01, 0],
                           [.15, .01, 0],
                           [.25, .01, 0],
                           [.35, .01, 0],
                           [.45, .01, 0],
                           [.55, .01, 0],

                           [.05, 2, 0],
                           [.15, 2, 0],
                           [.25, 2, 0],
                           [.35, 2, 0],
                           [.45, 2, 0],
                           [.55, 2, 0],

                           [-0.2, 2, 0],
                           [-0.1, 2, 0],

                           [.1, .01, 0],
                           [.2, .01, 0],
                           [.3, .01, 0],
                           [.4, .01, 0],
                           [.5, .01, 0],
                           [.6, .01, 0],

                           [.1, 2, 0],
                           [.2, 2, 0],
                           [.3, 2, 0],
                           [.4, 2, 0],
                           [.5, 2, 0],
                           [.6, 2, 0],

                           [0, 2, 0],
                           [0, .01, 0]])

    return train_data_file, real_f, test_u0, test_t, visualize


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, u0, dt, u1, layers, method, NUM):

        self.u0 = u0
        self.u1 = u1
        self.dt = dt

        self.layers = layers
        self.method = method
        self.NUM = NUM

        self.weights, self.biases = self.initialize_NN(layers)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        # tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # tfconfig.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=tfconfig)

        self.dt_tf = tf.placeholder(tf.float32, shape=[None, self.dt.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])

        self.f_pred = self.net_f(self.u0_tf)
        self.residual_pred = self.net_r(self.u0_tf, self.u1_tf, self.dt_tf)
        self.loss = tf.reduce_sum(tf.square( self.residual_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.train_op_Adam = tf.train.AdamOptimizer().minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        # H = 2.0 * (X - mylb) / (myub - mylb) - 1.0
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
            # H = tf.nn.dropout(H,0.8)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_f(self, u0):
        return self.neural_net(u0, self.weights, self.biases)

    def net_RK1(self, u0, dt):
        return u0 + dt * self.net_f(u0)

    def net_RK2(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + 1/2 * K1)
        return u0 + K2

    def net_Heun(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + K1)
        return u0 + 1/2 * K1 + 1/2 * K2

    # def net_Alpha(self, u0, dt):
    #     K1 = dt * self.net_f(u0)
    #     K2 = dt * self.net_f(u0 + )

    def net_RK3(self, u0, dt):  # SSPRK3
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + K1)
        K3 = dt * self.net_f(u0 + 1/4 * K1 + 1/4 * K2)
        return u0 + (K1 + K2 + 4 * K3)/6.

    def net_RK4(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + K1 / 2)
        K3 = dt * self.net_f(u0 + K2 / 2)
        K4 = dt * self.net_f(u0 + K3)
        return u0 + (K1 + 2*K2 + 2*K3 + K4)/6.

    def net_RK5(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + 1/2 * K1)
        K3 = dt * self.net_f(u0 + 1/4 * K1 + 1/4 * K2)
        K4 = dt * self.net_f(u0 - K2 + 2 * K3)
        K5 = dt * self.net_f(u0 + 7/27*K1+10/27*K2+1/27*K4)
        K6 = dt * self.net_f(u0 + 28/625*K1-1/5*K2+546/625*K3+54/625*K4-378/625*K5)
        return u0 + 1/24*K1+5/48*K4+27/56*K5+125/336*K6

    def net_RK7(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + 1/2 * K1)
        K3 = dt * self.net_f(u0 + 1/3 * K1)
        K4 = dt * self.net_f(u0 + 1/3 * K1 + 1/3 * K3)
        K5 = dt * self.net_f(u0 + 1/4 * K1)
        K6 = dt * self.net_f(u0 + 1/4 * K1 + 1/4 * K5)
        K7 = dt * self.net_f(u0 + 1/4 * K1 + 1/4 * K5 + 1/4 * K6)
        return u0 + 2 * K2 - 9/2 * K3 - 9/2 * K4 + 8/3 * K5 + 8/3 * K6 + 8/3 * K7

    def net_RK5X(self, u0, dt):
        K1 = dt * self.net_f(u0)
        K2 = dt * self.net_f(u0 + 1/4 * K1)
        K3 = dt * self.net_f(u0 + 1/8 * K1 + 1/8 * K2)
        K4 = dt * self.net_f(u0 - 1/2 * K2 + K3)
        K5 = dt * self.net_f(u0 + 3/16 * K1 + 9/16 * K4)
        K6 = dt * self.net_f(u0 - 3/7 * K1 + 2/7 * K2 + 12/7 * K3 - 12/7 * K4 + 8/7 * K5)
        return u0 + 1/90 * (7 * K1 + 32 * K3 + 12 * K4 + 32 * K5 + 7 * K6)

    def time_integrator_net(self, U, dt, method):
        if method == "RK1":
            return self.net_RK1(U, dt)
        elif method == "RK3":
            return self.net_RK3(U, dt)
        elif method == "RK4":
            return self.net_RK4(U, dt)
        elif method == "RK5":
            return self.net_RK5(U, dt)
        elif method == "RK5X":
            return self.net_RK5X(U, dt)
        elif method == "RK7":
            return self.net_RK7(U, dt)
        else:
            print("Error, method not found")
            return None

    def net_r(self, u0, u1, dt):
        U = u0
        for i in range(self.NUM):
            U = self.time_integrator_net(U, dt/self.NUM, self.method)
        return U - u1

    def callback(self, loss):
        print('Loss:', loss)

    def train_with_batchs(self, epochs, batch_size=32):
        for epoch in range(epochs):
            myu0 = tf.data.Dataset.from_tensor_slices(self.u0)
            mydt = tf.data.Dataset.from_tensor_slices(self.dt)
            myu1 = tf.data.Dataset.from_tensor_slices(self.u1)
            dcomb = tf.data.Dataset.zip((myu0, mydt, myu1)).shuffle(20000).batch(batch_size)
            iterator = dcomb.make_initializable_iterator()
            next_element = iterator.get_next()
            self.sess.run(iterator.initializer)

            try:
                while True:
                    [batch_u0, batch_dt, batch_u] = self.sess.run(next_element)
                    tf_dict = {self.u0_tf: batch_u0, self.dt_tf: batch_dt, self.u1_tf: batch_u}
                    self.sess.run(self.train_op_Adam, tf_dict)
            except tf.errors.OutOfRangeError:
                # print('End of Data', round_i,';Epoch:', it)
                pass

            if epoch % 50 == 0:
                tf_dict = {self.u0_tf: self.u0, self.dt_tf: self.dt, self.u1_tf: self.u1}
                loss_value = self.sess.run(self.loss, tf_dict)
                print('Epoch: %d, Loss: %.5e ' % (epoch, loss_value))

        tf_dict = {self.u0_tf: self.u0, self.dt_tf: self.dt, self.u1_tf: self.u1}
        self.optimizer.minimize(self.sess,
                                                feed_dict=tf_dict,
                                                fetches=[self.loss],
                                                loss_callback=None)  # self.callback)
        loss_value = self.sess.run(self.loss, tf_dict)
        print('Loss: %.5e' % (loss_value))

    def train(self, epochs, nIter, tol = 1.0e-5):
        flag = False
        tf_dict = {self.u0_tf: self.u0, self.dt_tf: self.dt, self.u1_tf: self.u1}
        loss_value = 1.0
        loop = 0
        while loop < epochs and (not flag):
            start_time = time.time()
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                if it % 1000 == 0:
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.5e, elapsed: %.2f ' % (it, loss_value, time.time()-start_time))
                    start_time = time.time()

                if loss_value < tol:
                    flag = True
                    break
            self.optimizer.minimize(self.sess,
                                            feed_dict=tf_dict,
                                            fetches=[self.loss],
                                            loss_callback=None)  # self.callback)
            loss_value = self.sess.run(self.loss, tf_dict)
            print('Epoch %d, with loss: %.5e, elapsed: %f ' % (loop, loss_value, time.time() - start_time))
            loop = loop + 1
        return flag

    def predict_f(self, X_star):
        F_star = self.sess.run(self.f_pred, {self.u0_tf: X_star})
        return F_star


def one_round(N_f, constant_dt, interval_number, method, model_name, train_method):

    if model_name == "Glycolytic":
        train_data_file, real_f, test_u0, test_t, visualize_function = generate_data_Glycolytic(N_f, constant_dt)
    elif model_name == "Cubic":
        train_data_file, real_f, test_u0, test_t, visualize_function = generate_data_Cubic(N_f, constant_dt)
    elif model_name == "Lorenz":
        train_data_file, real_f, test_u0, test_t, visualize_function = generate_data_Lorenz(N_f, constant_dt)
    elif model_name == "Hopf":
        train_data_file, real_f, test_u0, test_t, visualize_function = generate_data_Hopf(N_f, constant_dt)
    elif model_name == "Multiscale":
        train_data_file, real_f, test_u0, test_t, visualize_function = generate_data_Multiscale(N_f, constant_dt)
    else:
        return None

    # Load the data
    data = scipy.io.loadmat(train_data_file)
    u0 = data['u0']
    dt = data['dt'] #.flatten()[:,None]
    u1 = data['u1']
    net_dim = u0.shape[1]
    print('new_dim', net_dim)

    layers = [net_dim,128, 128, net_dim]  #network info (two hidden layer with 128 nodes)
    model = PhysicsInformedNN(u0, dt, u1, layers, method, interval_number)  # define the model

    if train_method == "batch":
        model.train_with_batchs(epochs=201, batch_size=32)  # train the model with batch data
    elif train_method == "full":
        model.train(epochs=1, nIter=30000)  # train the model with full data
    else:
        return None

    def learned_f(x, t):  # once the model is learnt, we obtain the learned_f
        f = model.predict_f(x[None, :])
        return f.flatten()

    # saver = tf.train.Saver()  # save the trained model into files
    # save_path = saver.save(model.sess, checkpoints_file)
    if model_name == "Hopf":
        learned_S = test_u0.shape[0]  # number of trajectories
        learned_N = test_t.shape[0]  # number of time snapshots
        learned_D = test_u0.shape[1]  # dimension

        learned_X_star = np.zeros((learned_S, learned_N, learned_D))
        real_X_star = np.zeros((learned_S, learned_N, learned_D))

        # solve ODE
        for k in range(0, learned_S):
            real_X_star[k, :, :] = odeint(real_f, test_u0[k, :], test_t)
            learned_X_star[k, :, :] = odeint(learned_f, test_u0[k, :], test_t)
        l2_error = np.linalg.norm(real_X_star.flatten()[:, None] - learned_X_star.flatten()[:, None], 2) / np.linalg.norm(real_X_star)
        visualize_function(test_t, real_X_star, learned_X_star)
    else:
        # next we will calculate the errors
        u_pred = odeint(learned_f, test_u0, test_t, hmax=1.0e-3)
        u_real = odeint(real_f, test_u0, test_t)
        l2_error = np.linalg.norm(u_pred - u_real, 2)/np.linalg.norm(u_real)
        visualize_function(test_t, u_real, u_pred)

    print('interval_number :',interval_number, '; L2_error',l2_error)
    return l2_error


if __name__ == "__main__":
    #model_name = "Cubic"
    #model_name = "Glycolytic"
    #model_name = "Lorenz"
    #model_name = "Hopf"
    model_name = "Multiscale"

    method = "RK4" #"RK5X" #"RK1"  # numerical scheme
    train_method =  "full" #"batch"
    result_file = 'Results_' + model_name + '_' + file_base_name_without_extension + '.txt'

    # N_f = 1000  # Number of sampling points
    # constant_dt = 0.1 # Time step size
    # interval_number = 20  # number of small intervals

    dict_Nf = [1000] #, 2000, 5000]
    #dict_dt = [0.1, 0.5, 1.0, 2.0, 4.0]
    dict_dt = [0.1, 0.25]
    #dict_dt = [0.0001, 0.001]
    dict_NUM = [20, 10, 1]

    for N_f in dict_Nf:
        for constant_dt in dict_dt:
            for interval_number in dict_NUM:
                print(N_f, constant_dt, interval_number, '\n')
                l2_error = one_round(N_f,constant_dt, interval_number, method,  model_name, train_method)
                R_file = open(result_file, "a")
                R_file.write(model_name + train_method + method + '; N_f' + str(N_f) + '; constant_dt' + str(constant_dt) + '; NUM:' + str(interval_number) + '; L2_error:' + str(l2_error) + '\n')
                R_file.close()
