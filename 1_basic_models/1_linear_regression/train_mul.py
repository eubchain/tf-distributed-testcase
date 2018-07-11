# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:17:49 2018

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
import numpy as np

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

      x_data=np.random.rand(10000) * 10
      y_data=0.5 * x_data + np.random.randn(10000) + 5
    
      x=tf.placeholder(dtype=tf.float32)
      y=tf.placeholder(dtype=tf.float32)
    
      w=tf.Variable(np.random.randn(),name='weights',dtype=tf.float32)
      b=tf.Variable(np.random.randn(),name='biases',dtype=tf.float32)
    
      predict=tf.multiply(w,x)+b
      loss=tf.reduce_mean(tf.square(predict-y))

      global_step = tf.contrib.framework.get_or_create_global_step()
      train_step=tf.train.AdamOptimizer(0.01).minimize(loss,global_step=global_step)

      init_op=tf.global_variables_initializer()

      sv = tf.train.Supervisor(
          is_chief=is_chief,
          init_op=init_op,
          global_step=global_step,  
          )
      
      sess_config = tf.ConfigProto(allow_soft_placement=True)
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
      
      step=0

      while not sv.should_stop() and step < 5000:
        _,step=sess.run([train_step, global_step],feed_dict={x:x_data,y:y_data})
        if step%100==0:
            print('Step %d: w=%f ,b=%f ,loss=%f'%(step,sess.run(w),sess.run(b),sess.run(loss,feed_dict={x:x_data,y:y_data})))
            

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
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

