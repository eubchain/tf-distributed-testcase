Implementing Multilayer Perceptron
=========
In this case we will implement linear regression as an iterative computational graph in TensorFlow. 

Data source
=========
The data set is about housing habitation, given some factors that affect the housing habitation and whether it is inhabited (dichotomous), such as lighting, temperature, etc.

Download data from the following url:

http://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

Model
=========
A multilayer perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

Output
=========
We print accuracy each 50 steps.

How to run
=========
For Single Computer:
---------
    python train.py

For Multiple Computers,use distributed code:
---------
With two steps:

1.run the following code with computer 192.168.1.173:2222

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/3_multilayer_prediction:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="ps"   --task_index=0

2.run the following code with computer 192.168.1.172:2222

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/3_multilayer_prediction:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="worker"   --task_index=0
