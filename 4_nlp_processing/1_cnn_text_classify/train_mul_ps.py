#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import tempfile
import shutil

import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("data_file", "./data/train-all.txt", "Data source for the training data.")
tf.flags.DEFINE_string("out_dir", "./output", "Data source for the training data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of hostname:port pairs')
tf.flags.DEFINE_string('worker_hosts', '','Comma-separated list of hostname:port pairs')
tf.flags.DEFINE_string('job_name', None, 'job name: worker or ps')
tf.flags.DEFINE_integer('task_index', None, 'Index of task within the job')
tf.flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))



def main(unused_argv):

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

    print ("After defining server")
    if FLAGS.job_name == 'ps':
        print("Parameter Server is executed")
        server.join()
    elif FLAGS.job_name == "worker":
        print("Parameter Server is executed")
        with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d"% FLAGS.task_index,
        cluster=cluster)):
            is_chief = (FLAGS.task_index == 0)
            # Data Preparation
            # ==================================================
            
            # Load data
            print("Loading data...")
            
            x_text, y_label = data_helpers.load_data_and_labels(FLAGS.data_file)
            
            # Build vocabulary
            max_document_length = max([len(x.split(" ")) for x in x_text])
            
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            x = np.array(list(vocab_processor.fit_transform(x_text)))
            y = np.array(y_label)
            
            # Randomly shuffle data
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(y)))
            print(type(x),type(y))
            
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
            
            # Split train/test set
            # TODO: This is very crude, should use cross-validation
            dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
            x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
            y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
            print (y_train.shape)
            print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
            print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
            
            # Training
            # ==================================================
            tf.MaxAcc = 0.1
            
            def copymax(path):
                shutil.copy(path, "{}.backup".format(path))
            
            
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            
            # Define Training procedure
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = FLAGS.out_dir
            print("Writing to {}\n".format(out_dir))
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            MaxAcc_prefi=os.path.join(checkpoint_dir, "MAXACCmodel")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            
            # Initialize all variables
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)

            init_op=tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir=out_dir,
                                     init_op=init_op,
                                     saver=saver,
                                     global_step=global_step
                                     )
            sess = sv.prepare_or_wait_for_session(server.target, config=session_conf)
            
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                _, current_step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict={cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.dropout_keep_prob: FLAGS.dropout_keep_prob})
                time_str = datetime.datetime.now().isoformat()
                if current_step % 100 ==0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")

                    loss, accuracy = sess.run(
                        [cnn.loss, cnn.accuracy],
                        feed_dict={cnn.input_x: x_batch,cnn.input_y: y_batch,cnn.dropout_keep_prob: 1.0})
                    time_str = datetime.datetime.now().isoformat()
                    result = "{}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy)
                    print(result)

                    with open(os.path.join(out_dir, "result"), 'a+') as f:
                        f.write("{}\n".format(result))

                    if tf.MaxAcc<accuracy:
                        tf.MaxAcc=accuracy
                        ifsave=True
                    else:
                        ifsave=False
                    print("Max acc:{}".format(tf.MaxAcc))

                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    if ifsave:
                        path = saver.save(sess, MaxAcc_prefi, None)
                        copymax("{}.data-00000-of-00001".format(path))
                        copymax("{}.index".format(path))
                        copymax("{}.meta".format(path))


if __name__ == '__main__':
    tf.app.run()