# cnn_text_classify

The base project is **[dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)**.  It can classify the positive and negative movie reviews. I modified it to have 5 labels(0,1,2,3,4).


It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.


## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Install tensorflow

Please see the [install guide](https://www.tensorflow.org/install/) in tensorflow website.


## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
-h, --help            show this help message and exit
--data_file DATA_FILE
File for training or evaluation (default: /data/train.txt for training, /data/dev.txt for evaluation)
--embedding_dim EMBEDDING_DIM
Dimensionality of character embedding (default: 128)
--filter_sizes FILTER_SIZES
Comma-separated filter sizes (default: '3,4,5')
--num_filters NUM_FILTERS
Number of filters per filter size (default: 128)
--l2_reg_lambda L2_REG_LAMBDA
L2 regularizaion lambda (default: 0.0)
--dropout_keep_prob DROPOUT_KEEP_PROB
Dropout keep probability (default: 0.5)
--batch_size BATCH_SIZE
Batch Size (default: 64)
--num_epochs NUM_EPOCHS
Number of training epochs (default: 100)
--evaluate_every EVALUATE_EVERY
Evaluate model on dev set after this many steps
(default: 100)
--checkpoint_every CHECKPOINT_EVERY
Save model after this many steps (default: 100)
--allow_soft_placement ALLOW_SOFT_PLACEMENT
Allow device soft device placement
--noallow_soft_placement
--log_device_placement LOG_DEVICE_PLACEMENT
Log placement of ops on devices
--nolog_device_placement

```

Train:

```bash
nvidia-docker run -it -p 2222:2222 -v /home/miner/tensorflow/cnn_text_classify:/root/code tensorflow/tensorflow:1.5.0-gpu  python ./code/train_mul.py --data_file=./code/data/train-all.txt --out_dir=./result --num_epochs=10 --ps_hosts="192.168.0.100:2222"   --worker_hosts="192.168.0.101:2222"   --job_name="ps"   --task_index=0
```

```bash
nvidia-docker run -it -p 2222:2222 -v /home/miner/tensorflow/cnn_text_classify:/root/code tensorflow/tensorflow:1.5.0-gpu  python ./code/train_mul.py --data_file=./code/data/train-all.txt --out_dir=./result --num_epochs=10  --ps_hosts="192.168.0.100:2222"   --worker_hosts="192.168.0.101:2222"   --job_name="worker"   --task_index=0
```

## Evaluating

```bash
./eval.py --checkpoint_file="./result/checkpoints/model-1400"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
