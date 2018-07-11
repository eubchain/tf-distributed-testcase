# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:46:04 2018

@author: eub_hmy
"""

import tensorflow as tf
import numpy as np

x1=np.random.rand(10000) * 10
y1=0.5 * x1 + np.random.randn(10000) + 5

x=tf.placeholder(dtype=tf.float32)
y=tf.placeholder(dtype=tf.float32)

w=tf.Variable(np.random.randn(),name='weights',dtype=tf.float32)
b=tf.Variable(np.random.randn(),name='biases',dtype=tf.float32)

predict=tf.multiply(w,x)+b
loss=tf.reduce_mean(tf.square(predict-y))
train_step=tf.train.AdamOptimizer(0.01).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for i in range(5000):
    sess.run(train_step,feed_dict={x:x1,y:y1})
    if i%100==0:
        print('Step %d: w=%f ,b=%f ,loss=%f'%(i,sess.run(w),sess.run(b),sess.run(loss,feed_dict={x:x1,y:y1})))


