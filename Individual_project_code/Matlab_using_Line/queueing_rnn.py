import keras
import keras.backend as k_back
from keras.constraints import *
import os
from keras.models import Model
import tensorflow as tf
from scipy.io import loadmat, savemat
from tensorflow.python.framework import tensor_util
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
import time
from collections import defaultdict
import datetime
from get_best_model import GetBestModel
from keras.initializers import RandomUniform, Constant
import numpy as np


class RNNCell(keras.layers.Layer):
    def __init__(self, trace, **kwargs):
        M = trace.size
        self.trace = k_back.constant(trace)
        self.index_jobs = slice(1, M+1)
        self.state_size = (M+1,)

        super().__init__(*kwargs)

    def build(self, shape_input):
        M = shape_input[1] - 1
        self.I = k_back.eye(M)
        init_mu = RandomUniform(minval=0.01, maxval=10)
        init_pfd = RandomUniform(minval=0.01, maxval=10)
        self.mu = self.add_weight('mu', shape=(M, 1), initializer=init_mu, constraint=NonNeg())
        data_p = self.add_weight('data_p', shape=(M, M-1), initializer=init_pfd, constraint=NonNeg())
        data_p_scaled = data_p/k_back.sum(data_p, axis=1, keepdims=True)
        self.P = k_back.reshape(k_back.flatten(data_p_scaled)[None, :] @ k_back.one_hot([j for j in range(M*M) if j % (M+1) != 0], M*M), (M, M))
        self.odot = (self.P - self.I)*self.mu
        self.is_built = True

    def init_xh(self, inputs):
        return_input = inputs
        return return_input, return_input

    def predict_xh(self, inputs, state):
        current_t = inputs[:, 0]
        old_t = state[:, 0]
        diff_t = current_t - old_t
        pred = state[:, self.index_jobs] + (diff_t[:, None]*k_back.minimum(state[:, self.index_jobs], self.trace)) @ self.odot
        pred_out = k_back.concatenate([current_t[:, None], pred], axis=1)
        return pred_out, pred_out

    def call(self, inputs, states):
        xh_first, st_first = self.init_xh(inputs)
        xh_pred, st_pred = self.predict_xh(inputs, states[0])
        decide_first = k_back.equal(inputs[:, 1], -1)
        xh = k_back.switch(decide_first, xh_pred, xh_first)
        st = k_back.switch(decide_first, st_pred, st_first)
        return xh, [st]

    def max_abs_percent_error(self, y_true, y_pred):
        y_true = k_back.mean(y_true, axis=0, keepdims=True)
        y_pred = k_back.mean(y_pred, axis=0, keepdims=True)
        self.y_true = y_true
        self.y_pree = y_pred
        ones_matrix = k_back.ones_like(y_true[:, :, self.index_jobs])
        zero_matrix = k_back.zeros_like(y_true[:, :, self.index_jobs])
        err = k_back.abs(y_true[:, :, self.index_jobs] - y_pred[:, :, self.index_jobs])*k_back.switch(
            k_back.equal(y_true[:, :, self.index_jobs], - ones_matrix), zero_matrix, ones_matrix)
        N = k_back.sum(y_true[:, :, self.index_jobs], axis=2)
        percentage_error = k_back.sum(err, axis=2)/(2*N)
        max_trace_error = k_back.max(percentage_error, axis=1)
        average_error = k_back.mean(max_trace_error, axis=0)
        return 100*average_error

    def loss(self, y_true, y_pred):
        return self.max_abs_percent_error(y_true, y_pred)

    def get_mu(self):
        return k_back.eval(self.mu)

    def get_p(self):
        return k_back.eval(self.P)

    def print(mu_P):
        print(self.get_mu())
        print(self.get_P())
        print(K.eval(self.odot))


class RNNCellTraining:
    def __init__(self, directory, rnn_cell):
        self.directory = directory
        self.rnn_cell = rnn_cell

    def read_files(self, fname):
        dic = list()
        with open(fname) as file:
            for line in file:
                toks = line.strip().split()
                dic += [float(i) for i in toks]
        return np.array(dic)

    def load_file(self, maxR=None, Hmax=None):
        list_traces = []
        list_inputs = []

        Hmin = Hmax
        for file in os.listdir(self.directory):
            if os.path.isfile(os.path.join(self.directory, file)) and file.endswith('.srv'):
                print(f'Loading definition of server concurrency')
                self.init_s = self.read_files(os.path.join(self.directory, file))
            if os.path.isfile(os.path.join(self.directory, file)) and file.endswith('.mat') and len(list_traces) != maxR:
                print(f'Loading the trace {file}')
                ml = loadmat(os.path.join(self.directory, file))
                trace_in = ml['average_queue_length_trace']
                trace_seq = trace_in[:, 0:1]
                H = trace_in.shape[0]
                print(H)
                if Hmin is None or Hmin > H:
                    Hmin = H
                input0 = trace_in[0, 1:]
                inputH = -np.ones((H-1, input0.shape[0]))
                input0H = np.concatenate([input0[np.newaxis], inputH])
                input_trace = np.hstack([trace_seq, input0H])
                list_traces.append(trace_in)
                list_inputs.append(input_trace)

        for i in range(len(list_traces)):
            print(Hmin)
            list_inputs[i] = list_inputs[i][:Hmin, :]
            list_traces[i] = list_traces[i][:Hmin, :]
        self.traces = np.stack(list_traces)
        self.inputs = np.stack(list_inputs)

        print('Loading Has Ended')

    def makeNN(self, lr):
        print('Building the RNN')
        self.cell = self.rnn_cell(self.init_s)
        tests_in = keras.Input((None, self.traces.shape[2]))
        rnn_layer = keras.layers.RNN(self.cell, return_sequences=True)
        rec = rnn_layer(tests_in)
        optimizer = keras.optimizers.Adam(lr=lr)
        self.model = Model(inputs=[tests_in], outputs=[rec])
        self.model.compile(optimizer=optimizer, metrics=[], loss=self.cell.loss)

    def learn(self):
        print('Start the Learning')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.01)
        h = keras.callbacks.History()
        gb = GetBestModel(monitor='val_loss', verbose=1, mode='min', period=1)
        self.learn_begin = time.time()
        hist = self.model.fit(self.inputs, self.traces, epochs=100000000, batch_size=1, validation_split=0.5, callbacks=[gb, early_stop, h])
        self.learn_end = time.time()
        self.val_loss = hist.history['val_loss'][-1]
        print('Learning has Ended')

    def saveResults(self, fname):
        now = datetime.datetime.now()
        print(f"Saving results on {fname}")
        mu = self.cell.get_mu()
        p = self.cell.get_p()
        with open(fname, "w") as f:
            print(f"tbeg {self.learn_begin}", file=f)
            print(f"tend {self.learn_end}", file=f)
            print(f"elapsed {self.learn_end-self.learn_begin}", file=f)
            print(f"val_loss {self.val_loss}", file=f)
            print(f"mu {' '.join([str(i[0]) for i in mu])}", file=f)
            print(f"P {' '.join([str(j) for i in p for j in i ])}", file=f)
