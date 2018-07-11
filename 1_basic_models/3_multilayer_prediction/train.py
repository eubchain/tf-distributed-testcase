# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:37:15 2018

@author: eub_hmy
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv("occupancy_data/datatest.txt")
            
X_train, X_test, y_train, y_test = train_test_split(\
data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,\
data["Occupancy"].values.reshape(-1, 1), random_state=2972,test_size=0.25)

y_train=np.concatenate((1-y_train,y_train),axis=1)
y_test=np.concatenate((1-y_test,y_test),axis=1)
training_steps=2000
learning_rate=0.01
batch_size=300

n_input=len(X_train[0])
n_class=len(y_train[0])
samples_num=len(y_train)
batch_num=int(samples_num/batch_size)

n_hidden=8

x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_class])
keep_prob=tf.placeholder(tf.float32)

w1=tf.Variable(np.random.rand(n_input,n_hidden),dtype=tf.float32)
b1=tf.Variable(np.random.rand(n_hidden),dtype=tf.float32)

w2=tf.Variable(np.random.rand(n_hidden,n_class),dtype=tf.float32)
b2=tf.Variable(np.random.rand(n_class),dtype=tf.float32)

h1=tf.nn.relu(tf.matmul(x,w1)+b1)
h1_drop=tf.nn.dropout(h1,keep_prob)

pred=tf.matmul(h1_drop,w2)+b2
cross_entropy=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for i in range(training_steps):
    batch_num=int(samples_num/batch_size)
    for j in range(batch_num):
        feed={x:X_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size],keep_prob:0.5}
        _,loss=sess.run([train_step,cross_entropy],feed_dict=feed)
    if i % 50 == 0:
        print("Step %d: accuracy=%f"%(i,sess.run(accuracy,feed_dict={x:X_test,y:y_test,keep_prob:1})))
sess.close()



