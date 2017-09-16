# mbari-tensorflow-detection
 
Code for testing Tensorflow detection API on underwater imagery

## Prerequisites
 
- Python version  3.5.4, on a Mac OSX download and install from here:
 https://www.python.org/downloads/mac-osx/ 

## Running

### Check-out the code

    $ git clone https://github.com/danellecline/mbari-tensorflow-detection

### Create virtual environment with correct dependencies, install tensorflow

    $ cd mbari-tensorflow-detection
    $ pip3 install virtualenv
    $ virtualenv --python=/usr/local/bin/python3.5 venv-pam
    $ source venv-mbari-tensorflow-detection/bin/activate
    $ pip3 install -r requirements.txt
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL
    
### Install Tensorflow models and object detection protocols
    $ git clone https://github.com/tensorflow/models.git tensorflow_models
    $ cd tensorflow_models
    $ protoc object_detection/protos/*.proto --python_out=.
    $ cd .. 

### Add libraries to PYTHONPATH

When running locally, the tensorflow_models directories should be appended to PYTHONPATH. 
This can be done by running the following from tensorflow_models :

    $ cd tensorflow_models
    $ export PYTHONPATH=$PYTHONPATH:`pwd`
    $ cd ..
    
### Generate the TFRecord files

    $ wget URL_FOR_TRAINING_DATA
    $ python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_train.record --label_map_path  mbari_benthic_label_map.pbtxt --set train 
    $ python create_tfrecord.py  
    --data_dir PATH_TO_TRAINING_DATA --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_test.record --label_map_path  mbari_benthic_label_map.pbtxt --set test 
    
### Train the model 
     
    $ python tensorflow_models/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=`pwd`/models/faster_rcnn_resnet50_coco.config \
    --checkpoint_dir=PATH_TO_TRAINING_DATA \
    --eval_dir=PATH_TO_EVAL_DIR

    
* Train model steps
* Test model steps

## Developer Notes

A placeholder for notes that might be useful for developers
 
* Install your own dataset [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)
* Install TensorFlow Object Detection API [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) 
* Running in the cloud [https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine](https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine)
* Configuring option detection pipeline [https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md)