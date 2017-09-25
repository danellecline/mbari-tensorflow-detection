# mbari-tensorflow-detection
 
Code for testing Tensorflow detection API on underwater imagery

## Prerequisites
 
- Python version  3.5, on a Mac OSX download and install from here:
 https://www.python.org/downloads/mac-osx/ 

## Running

### Check-out the code

    $ git clone https://github.com/danellecline/mbari-tensorflow-detection

### Create virtual environment with correct dependencies

    $ cd mbari-tensorflow-detection
    $ pip3 install virtualenv
    $ virtualenv --python=/usr/local/bin/python3.5 venv-mbari-tensorflow-detection
    $ source venv-mbari-tensorflow-detection/bin/activate
    $ pip3 install -r requirements.txt
    
### Install Tensorflow for Mac OSX
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL
    
### Install Tensorflow for Ubuntu GPU
Also see [https://www.tensorflow.org/install/install_linux](https://www.tensorflow.org/install/install_linux)

    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl 
    $ pip3 install --upgrade tensorflow-gpu==1.3.0
    
### Install Tensorflow models and object detection protocols
    $ git clone https://github.com/tensorflow/models.git tensorflow_models
    $ cd tensorflow_models
    $ protoc object_detection/protos/*.proto --python_out=.
    $ cd .. 

### Add libraries to PYTHONPATH

When running locally, the tensorflow_models directories should be appended to PYTHONPATH. 
This can be done by running the following from tensorflow_models :

    $ cd tensorflow_models
    $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    $ cd ..
    
### Generate the TFRecord files

    $ wget URL_FOR_TRAINING_DATA
    $ python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_train.record --label_map_path  mbari_benthic_label_map.pbtxt --set train 
    $ python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_test.record --label_map_path  mbari_benthic_label_map.pbtxt --set test 
    
### Edit the pipeline.config file
Insert the correct paths for the training/test data in PATH_TO_BE_CONFIGURED 

### Train the model 
     
    $ python tensorflow_models/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=`pwd`/models/faster_rcnn_resnet50_coco/pipeline.config \ 
    --train_dir=`pwd`/models/faster_rcnn_resnet50_coco/checkpoints \ 
    --eval_dir=`pwd`/models/faster_rcnn_resnet50_coco/eval
      
### Test the model 

    $ python tensorflow_models/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=`pwd`/models/faster_rcnn_resnet50_coco/pipeline.config \ 
    --checkpoint_dir=`pwd`/models/checkpoints/ \
    --eval_dir=PATH_TO_EVAL_DIR

## Bug fix
add to tensorflow_models/object_detection/core/preprocessor.py 
mean = list(mean) before lines 1474.
https://www.bountysource.com/issues/48005318-bug-change-protobuf-to-list-in-object-detection-api

## Developer Notes

TF_CONFIG - environment variable

A placeholder for notes that might be useful for developers
* Pre processing options [https://github.com/tensorflow/models/blob/master/object_detection/protos/preprocessor.proto](https://github.com/tensorflow/models/blob/master/object_detection/protos/preprocessor.proto) 
* Install your own dataset [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)
* Install TensorFlow Object Detection API [dhttps://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.m](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) 
* Running in the cloud [https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine](https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine)
* Configuring option detection pipeline [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md)
* To see GPU usage on DevBox 
    $ watch nvidia-smi 
* To run train/eval on different GPUS, add to train/eval.py
  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" or "1" # to use first or second GPU
* How to see more boxes in Tensorboard [https://stackoverflow.com/questions/45452376/small-object-detection-with-faster-rcnn-in-tensorflow-models](https://stackoverflow.com/questions/45452376/small-object-detection-with-faster-rcnn-in-tensorflow-models)
* Good overview article on the different detection methods [https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9](https://medium.com/towards-data-science/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9) 