# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:52:14 2018

@author: eub_hmy
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import os
tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 1000000     # train step 上限
batch_size = 512            
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)
num_checkpoints=5
checkpoint_every=100

MODEL_DIR = "model/ckpt"
#模型保存的名字
MODEL_NAME = "model.ckpt"

if not tf.gfile.Exists(MODEL_DIR):
    tf.gfile.MakeDirs(MODEL_DIR)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs],name='input')
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
prediction=tf.nn.softmax(pred,name='softmax')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=num_checkpoints)

sess=tf.Session()
sess.run(init)
step = 1
while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    if step % 20 == 0:
        print(step,sess.run(accuracy, feed_dict={
        x: batch_xs,
        y: batch_ys,
    }))
    sess.run([train_op], feed_dict={
        x: batch_xs,
        y: batch_ys,
    })
    if step % checkpoint_every == 0:
        saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME),global_step=step)
    step += 1
    