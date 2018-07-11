Using the TensorFlow Way of Regression
=========
In this case we will implement linear regression as an iterative computational graph in TensorFlow. 

Data source
=========
We randomly generate 10000 sets of data,with the formula y = 0.5 * x + 5.

Model
=========
The the output of our model is a 2D linear regression:

    y = W * x + b

Output
=========
We print W and b value each 100 steps.

How to run
=========
For Single Computer:
---------
    python train.py

For Multiple Computers,use distributed code:
---------
With two steps:

1.run the following code with computer 192.168.1.173:2222

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/1_linear_regression:/root/code tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="ps"   --task_index=0

2.run the following code with computer 192.168.1.172:2222

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/1_linear_regression:/root/code tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="worker"   --task_index=0
