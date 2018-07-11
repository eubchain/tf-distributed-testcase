# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:02:05 2018

@author: eub_hmy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    num_worker = len(worker_hosts)
    print ("Number of worker = " + str(num_worker))
    print ("ps_hosts = ")
    print(*ps_hosts)
    print ("worker_hosts = ")
    print(*worker_hosts)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts,"worker": worker_hosts})

    print ("After defining Cluster")
    print ("Job name = " + FLAGS.job_name)
    print ("task index = " + str(FLAGS.task_index))

    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    print ("After defining server")

    if FLAGS.job_name =="ps":
        server.join()
    elif FLAGS.job_name =="worker":
        with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d"% FLAGS.task_index,
        cluster=cluster)):

            is_chief = (FLAGS.task_index == 0)
            data=pd.read_csv(FLAGS.data_dir)
            
            X_train, X_test, y_train, y_test = train_test_split(\
            data[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]].values,\
            data["Occupancy"].values.reshape(-1, 1), random_state=42,test_size=0.25)
            
            y_train=np.concatenate((1-y_train,y_train),axis=1)
            y_test=np.concatenate((1-y_test,y_test),axis=1)
            
            training_steps=FLAGS.training_steps
            learning_rate=FLAGS.learning_rate
            batch_size=FLAGS.train_batch_size
            features=len(X_train[0])
            class_num=len(y_train[0])
            samples_num=len(y_train)
            batch_num=int(samples_num/batch_size)
            
            x=tf.placeholder(tf.float32,[None,features])
            y=tf.placeholder(tf.float32,[None,class_num])
            
            w=tf.Variable(np.random.rand(features,class_num),dtype=tf.float32)
            b=tf.Variable(np.random.rand(class_num),dtype=tf.float32)
            
            pred=tf.matmul(x,w)+b
            cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)
            
            correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
            accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
            
            init=tf.global_variables_initializer()
            
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                init_op=init,
                global_step=global_step,  
                )
            
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
            
            step=0
            while not sv.stop() and step <= training_steps:
                i=random.randrange(batch_num)
                feed={x:X_train[i*batch_size:(i+1)*batch_size],y:y_train[i*batch_size:(i+1)*batch_size]}
                _,loss,step=sess.run([train_step,cost,global_step],feed_dict=feed)
                if step % 100 == 0:
                    print("Step %d: accuracy=%f"%(step,sess.run(accuracy,feed_dict={x:X_test,y:y_test})))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='input/occupancy_data/datatest.txt',
        help="""\
        Where to download the speech training data to.
        """)
    parser.add_argument(
        '--training_steps',
        type=int,
        default=3000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=1000,
        help='How many lines to train on at a time.'
    )
    parser.add_argument(
        '--task_index',
        type=int,
        default=0,
        help="""\
        Index of task within the job.\
        """
    )
    parser.add_argument(
        '--ps_hosts',
        type=str,
        default=0,
        help="""\
        Comma-separated list of hostname:port pairs.\
        """
    )
    parser.add_argument(
        '--worker_hosts',
        type=str,
        default=0,
        help="""\
        Comma-separated list of hostname:port pairs.\
        """
    )
    parser.add_argument(
        '--job_name',
        type=str,
        default=0,
        help="""\
        job name: worker or ps.\
        """
    )
    parser.add_argument(
        '--issync',
        type=int,
        default=0,
        help="""\
        between graph or not.\
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)