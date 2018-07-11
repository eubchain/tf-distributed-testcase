# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:14:25 2018

@author: eub_hmy
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os

tf.flags.DEFINE_string("data_dir", "./data/dev.txt", "Data source for the evaluation data.")

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_file", "", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("train_batch_size", 512, "The training batch Size (default: 512)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

rootdir = FLAGS.data_dir
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
print("开始将图片转化为数组")
result = np.array([]) #创建一个空的一维数组
for i in range(0,len(list)):
    im = Image.open(os.path.join(rootdir,list[i])).convert('L')
    im=im.resize((28, 28),Image.ANTIALIAS)
    im = np.asarray(im)
    if im.mean()>100:
        im=255-im
    result=np.append(result,im)

im=np.zeros((FLAGS.train_batch_size-len(list),28,28))
im = np.asarray(im) 
result=np.append(result,im)

result=result.reshape(-1,28,28)
result=result/256

checkpoint_file = FLAGS.checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file),clear_devices=True)
        saver.restore(sess, checkpoint_file)

        input_x = sess.graph.get_tensor_by_name("input:0")
        out_softmax = sess.graph.get_tensor_by_name("softmax:0")

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x:result})
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        print("label:", prediction_labels[:len(list)])
        print(list)
