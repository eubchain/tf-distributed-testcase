# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 11:52:14 2018

@author: eub_hmy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import sys
import os


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuid
    print ("Inside main function")
    # Setup the directory we'll write summaries to for TensorBoard
  
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print ('job_name : ' + FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print ('task_index : ' + str(FLAGS.task_index))
  
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
  
    # 创建集群
    num_worker = len(worker_spec)
    print ("Number of worker = " + str(num_worker))
    print ("ps_spec = ")
    print(*ps_spec)
    print ("worker_spec = ")
    print(*worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    print ("After defining Cluster")
    print ("Job name = " + FLAGS.job_name)
    print ("task index = " + str(FLAGS.task_index))
    # try:
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # except:
    #     print ("Unexpected error:" + sys.exc_info()[0])
    print ("After defining server")
    if FLAGS.job_name == 'ps':
        print("Parameter Server is executed")
        server.join()
    elif FLAGS.job_name == "worker":
        print("Parameter Server is executed")

        is_chief = (FLAGS.task_index == 0)
        # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d"% FLAGS.task_index,
        cluster=cluster)):
            # 导入数据
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        
            # hyperparameters
            training_iters = FLAGS.how_many_training_steps     # train step 上限
            batch_size = FLAGS.batch_size            
            n_inputs = 28               # MNIST data input (img shape: 28*28)
            n_steps = 28                # time steps
            n_hidden_units = 128        # neurons in hidden layer
            n_classes = 10              # MNIST classes (0-9 digits)
            MODEL_DIR = FLAGS.model_dir
            MODEL_NAME = "model.ckpt"
            num_checkpoints=FLAGS.num_checkpoints
        
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
            global_step = tf.train.get_or_create_global_step()
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost,global_step=global_step)
        
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
            saver = tf.train.Saver(max_to_keep=num_checkpoints)
            init = tf.global_variables_initializer()
            
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                init_op=init,
                saver=saver,
                global_step=global_step,  
                )
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    
            step = 0
            while step < training_iters:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                _,loss,step=sess.run([train_op,cost,global_step], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                if step % 20 == 0:
                    print("Step %d: accuracy=%f"%(step,sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })))
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess,os.path.join(MODEL_DIR, MODEL_NAME), global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                

if __name__ == '__main__':
    print ("Executing main function")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='model/ckpt',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=1000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--num_checkpoints',
        type=int,
        default=5,
        help='How many ckpts to keep.'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=100,
        help='Save model after this many steps (default: 100)'
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
        default='',
        help="""\
        Comma-separated list of hostname:port pairs.\
        """
    )
    parser.add_argument(
        '--worker_hosts',
        type=str,
        default='',
        help="""\
        Comma-separated list of hostname:port pairs.\
        """
    )
    parser.add_argument(
        '--job_name',
        type=str,
        default='',
        help="""\
        job name: worker or ps.\
        """
    )
    parser.add_argument(
        '--issync',
        type=int,
        default='0',
        help="""\
        是否采用分布式的同步模式，1表示同步模式，0表示异步模式.\
        """
    )
    parser.add_argument(
        '--gpuid',
        type=str,
        default='',
        help="""\
        use which gpu\
        """
    )
  
    FLAGS, unparsed = parser.parse_known_args()
  
    # # # Test printing all arguments
    # print ("worker_hosts = " + str(FLAGS.worker_hosts))
    # print ("ps_hosts = " + FLAGS.ps_hosts)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
