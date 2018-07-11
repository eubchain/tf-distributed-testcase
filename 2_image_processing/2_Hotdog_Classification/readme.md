Hotdog-Classification
=========
A simple TensorFlow app to classify whether a supplied image is a hotdog or not.
Source for post on medium :

https://medium.com/@faze.php/using-tensorflow-to-classify-hotdogs-8494fb85d875


Buy me a cup of coffee <3

1LHT8uYsmQW8rCQjx58CJVZLgGxKL5pL89

^ Bitcoin RULZ ALL


Command to run Cluster
=========
PS node:
    nvidia-docker run -itd -p 2222:2222 -v /root:/notebooks tensorflow/tensorflow:1.7.0-gpu python ./code/retrain.py   --bottleneck_dir=./bottlenecks   --model_dir=./inception   --summaries_dir=./result/training_summaries/long   --output_graph=./result/retrained_graph.pb   --output_labels=./result/retrained_labels.txt   --image_dir=./input/images   --job_name="worker"   --ps_hosts="192.168.0.100:2222" --worker_hosts="192.168.0.101:2222" --task_index=0

worker node:
    nvidia-docker run -itd -p 2222:2222 -v /root:/notebooks tensorflow/tensorflow:1.7.0-gpu python ./code/retrain.py   --bottleneck_dir=./bottlenecks   --model_dir=./inception   --summaries_dir=./result/training_summaries/long   --output_graph=./result/retrained_graph.pb   --output_labels=./result/retrained_labels.txt   --image_dir=./input/images   --job_name="ps"   --ps_hosts="192.168.0.100:2222" --worker_hosts="192.168.0.101:2222" --task_index=0