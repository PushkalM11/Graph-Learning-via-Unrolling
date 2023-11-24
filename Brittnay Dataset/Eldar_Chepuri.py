import numpy as np
from numpy import linalg as LA
from numpy.linalg import matrix_power as MP

import random
import scipy
from scipy.spatial.distance import cdist
from scipy.io import savemat, loadmat

import networkx as nx
import sys

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.layers import Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping

print(f"TF Version: {tf.__version__}")

def adj_to_laplacian(AdjL):
    tmp = np.array(AdjL, dtype = "float32")
    return np.diag(np.sum(tmp, axis = 0)) - tmp

X_train = np.array(loadmat('./Dataset/X_train.mat')['X'])

Laplacian = np.array(loadmat('./Dataset/Chepuri.mat')['Ls_noiseless'], dtype = "float")

N, T = X_train.shape

dt = np.zeros([T - 1, T])

for i in range(T - 1):
    dt[i, i] = -1
    dt[i, i - 1] = 1

d = dt.T
d = np.concatenate((d, np.zeros([T, 1])), axis = 1) 
d[T - 1, T - 1] = -1
d = d.T
d[T - 1, T - 1] = 1 
d[-1, -2] = 1
d[0, -1] = 0
d = tf.convert_to_tensor(d, dtype = "float32")
print(d)

psi_cost = 0
test_list = []

# A - Predicted, B - Target matrix
def costfunc(A, B):
    return (tf.norm(tf.multiply(psi_cost, (A - B)))) ** 2

def mse_unknown(A, B):

    global psi_cost
    psi = psi_cost

    psid = tf.convert_to_tensor(np.ones([N, T], dtype = float) - psi, dtype = "float32")
    mse_unk = (LA.norm(np.multiply(psid, (X_train - B)))) ** 2 / np.sum(psid.numpy().flatten())
    
    global mse_unknown_list
    mse_unknown_list.append(mse_unk)
    
    return mse_unk

def mse_known(A, B):

    global psi_cost 
    psi = psi_cost

    mse_kn = (LA.norm(np.multiply(psi, (X_train - B)))) ** 2 / np.sum(psi.numpy().flatten())
    
    global mse_known_list
    mse_known_list.append(mse_kn)

    return mse_kn

callb = EarlyStopping(monitor = 'loss', mode = 'min', min_delta = 0.00001, verbose = 1, patience = 10)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr / lr_decay_factor

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

class EldarModel(tf.keras.Model):
    
    def __init__(self, Laplacian, reg_init, loops_init, d_init, **kwargs):
        super(EldarModel, self).__init__(**kwargs)
        p_init = 15.0
        self.p1_init = tf.constant(p_init, shape = (1, 1), dtype = "float32")
        self.p1 = tf.Variable(initial_value = self.p1_init, trainable = True, name = "p1")
        self.p2_init = tf.constant(p_init, shape = (1, 1), dtype = "float32")
        self.p2 = tf.Variable(initial_value = self.p2_init, trainable = True, name = "p2")
        self.p3_init = tf.constant(p_init, shape = (1, 1), dtype = "float32")
        self.p3 = tf.Variable(initial_value = self.p3_init, trainable = True, name = "p3")
        self.p4_init = tf.constant(p_init, shape = (1, 1), dtype = "float32")
        self.p4 = tf.Variable(initial_value = self.p4_init, trainable = True, name = "p4")
        
        self.lap = Laplacian
        self.reg = reg_init
        self.loops = loops_init
        self.d_matrix = d_init
        
    def call(self, y):
        
        psi = y[0, :, T:]
        global psi_cost
        psi_cost = psi
        
        psi = tf.convert_to_tensor(psi, dtype = "float32")
        
        X_in = y[0, :, : T]
        X_in = tf.convert_to_tensor(X_in, dtype = "float32")
        
        Y = X_in
        
        LG, reg, loop, d = self.lap, self.reg, self.loops, self.d_matrix
        
        LT = tf.matmul(tf.transpose(d), d)
        LT = tf.convert_to_tensor(LT, dtype = "float32")
        
        HLG = tf.eye(tf.shape(LG)[0], dtype = "float32") 
        GLT = LT + (self.p1 * MP(LT, 2)) + (self.p2 * MP(LT, 3)) + (self.p3 * MP(LT, 4)) + (self.p4 * MP(LT, 5))
        
        Xk = tf.zeros_like(X_in)
        Zk = tf.zeros_like(X_in)
        Zk = -(tf.multiply(psi, Xk) - Y + (reg * (HLG @ Xk @ GLT)))
        
        for i in range(loop):
            fdx_xk = tf.multiply(psi, Xk) - Y + (reg * (HLG @ Xk @ GLT))
            fdx_zk = tf.multiply(psi, Zk) - Y + (reg * (HLG @ Zk @ GLT))
            
            tau = tf.linalg.trace(tf.transpose(fdx_xk) @ Zk) / tf.linalg.trace(tf.transpose((Y + fdx_zk)) @ Zk)
            
            Xk_1 = Xk - (tau * Zk)
            
            fdx_xk_1 = tf.multiply(psi, Xk_1) - Y + (reg * (HLG @ Xk_1 @ GLT))
            
            gamma = (tf.norm(fdx_xk_1) ** 2) / (tf.norm(fdx_xk) ** 2)
            
            Zk_1 = (gamma * Zk) - fdx_xk_1
            
            Xk = Xk_1
            Zk = Zk_1
        
        global test_list
        test_list.append(Xk_1.numpy())
        
        return tf.reshape(Xk_1, [1, N, T])


no_of_psi = 10
sensing_ratio = np.arange(1.0, 81.0, 1.0) / 100.0

reg_list = [5.0e-6] * len(sensing_ratio)
loops_list = [50] * len(sensing_ratio)

all_psi = []

mse_known_list = []
mse_unknown_list = []

learned_variables = []

lr_decay_factor = 1.06

for i_sen, rem in enumerate(sensing_ratio):
    
    print(f"\n\n###########################################################################################")
    print(f"################################### Sensing Ratio: {rem * 100}% ###################################")
    print(f"###########################################################################################\n\n")
    
    if i_sen == 0:
        M = int(rem * T)
        X_train_missing = np.zeros([no_of_psi, N, T])
        X_train_concatenated = np.zeros([no_of_psi, N, 2*T])
    
        for i in range(no_of_psi):
            psi_k = np.array([0] * (N * M) + [1] * (N * (T - M)))
            np.random.shuffle(psi_k)
            psi_k = psi_k.reshape([N, T])
            all_psi.append(psi_k)
            X_train_missing[i, : , : ] = X_train * psi_k
            X_train_concatenated[i, :, :] = np.concatenate((X_train_missing[i], psi_k), axis = 1)
    
    else:
        previous_psi = all_psi[-no_of_psi : ]
        X_train_missing = np.zeros([no_of_psi, N, T])
        X_train_concatenated = np.zeros([no_of_psi, N, 2 * T])
        
        rem_diff = int(N * (rem - sensing_ratio[i_sen - 1]) * T)
        
        for i in range(no_of_psi):
            psi_k_tmp = np.array(previous_psi[i]).flatten()
            ones_idx = np.where(psi_k_tmp == 1)
            
            ones_to_zero_idx = np.random.choice(ones_idx[0], size = rem_diff)
            psi_k_tmp[ones_to_zero_idx] = 0
            psi_k = psi_k_tmp
            
            psi_k = psi_k.reshape([N, T])
            all_psi.append(psi_k)
            X_train_missing[i, : , : ] = X_train * psi_k
            X_train_concatenated[i, :, :] = np.concatenate((X_train_missing[i], psi_k), axis = 1)
        
    reg_init = reg_list[i_sen]
    loops_init = loops_list[i_sen]
    
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate = 1.0e-2, momentum = 0.001)
    
    eldar_model = EldarModel(Laplacian, reg_init, loops_init, d, name = 'EM')
    eldar_model.compile(optimizer = sgd_optimizer, loss = costfunc, 
                        metrics = [mse_known, mse_unknown], run_eagerly = True)
    eldar_model.fit(X_train_concatenated, X_train_missing, epochs = 100, 
                    callbacks = [lr_scheduler], batch_size = 1)
    eldar_model.summary()
    
    all_variables = eldar_model.variables
    
    graph_vars = [i.numpy()[0, 0] for i in all_variables]
    learned_variables.append(graph_vars)
    
    print(f"Learned Variables: {graph_vars}")

for i, rem in enumerate(sensing_ratio):
    print(f"For {int(100 * rem)}% sensing ratio: {learned_variables[i]}")

np.save('./Outputs/eldar_chepuri_mse_known.npy', mse_known_list)
np.save('./Outputs/eldar_chepuri_mse_unknown.npy', mse_unknown_list)