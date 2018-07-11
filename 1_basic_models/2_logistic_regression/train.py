# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:02:05 2018

@author: eub_hmy
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv("occupancy_data/datatest.txt")

X_train, X_test, y_train, y_test = train_test_split(\
data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,\
data["Occupancy"].values.reshape(-1, 1), random_state=42,test_size=0.25)

y_train=np.concatenate((1-y_train,y_train),axis=1)
y_test=np.concatenate((1-y_test,y_test),axis=1)

training_steps=1000
learning_rate=0.0001
batch_size=500
features=len(X_train[0])
class_num=len(y_train[0])
samples_num=len(y_train)


x=tf.placeholder(tf.float32,[None,features])
y=tf.placeholder(tf.float32,[None,class_num])

w=tf.Variable(np.random.rand(features,class_num),dtype=tf.float32)
b=tf.Variable(np.random.rand(class_num),dtype=tf.float32)

pred=tf.matmul(x,w)+b
cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))

train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for i in range(training_steps):
    batch_num=int(samples_num/batch_size)
    for j in range(batch_num):
        feed={x:X_train[j*batch_size:(j+1)*batch_size],y:y_train[j*batch_size:(j+1)*batch_size]}
        _,loss=sess.run([train_step,cost],feed_dict=feed)
    if i % 50 == 0:
        print("Step %d: accuracy=%f"%(i,sess.run(accuracy,feed_dict={x:X_test,y:y_test})))
sess.close()