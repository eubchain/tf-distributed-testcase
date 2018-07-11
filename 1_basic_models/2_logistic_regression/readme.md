Implementing Logistic Regression
=========
Logistic regression is a way to predict a number between zero or one (usually we consider the output a probability). This prediction is classified into class value ‘1’ if the prediction is above a specified cut off value and class ‘0’ otherwise.  The standard cutoff is 0.5.  For the purpose of this example, we will specify that cut off to be 0.5, which will make the classification as simple as rounding the output.

Data source
=========
The data set is about housing habitation, given some factors that affect the housing habitation and whether it is inhabited (dichotomous), such as lighting, temperature, etc.

Download data from the following url:

http://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

Model
=========
The the output of our model is the standard logistic regression:

y = sigmoid(A * x + b)

The x matrix input will have dimensions (batch size x # features).  The y target output will have the dimension batch size x 1.

The loss function we will use will be the mean of the cross-entropy loss:

loss = mean( - y * log(predicted) + (1-y) * log(1-predicted) )

TensorFlow has this cross entropy built in, and we can use the function, 'tf.nn.softmax\_cross\_entropy\_with\_logits\_v2()'

We will then iterate through random batch size selections of the data.

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

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/2_logistic_regression:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="ps"   --task_index=0

2.run the following code with computer 192.168.1.172:2222

    nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/2_logistic_regression:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="worker"   --task_index=0
