# mbari-tensorflow-detection
 
Code for testing Tensorflow detection API on underwater imagery

## Prerequisites
 
- Python version  3.5.4, on a Mac OSX download and install from here:
 https://www.python.org/downloads/mac-osx/ 

## Running

Check-out the code

    $ git clone https://github.com/danellecline/mbari-tensorflow-detection

Create virtual environment with correct dependencies, install tensorflow, and object detection protocols

    $ pip3 install virtualenv
    $ virtualenv --python=/usr/local/bin/python3.5 venv-pam
    $ source venv-mbari-tensorflow-detection/bin/activate
    $ pip3 install -r requirements.txt
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL
    $ git clone https://github.com/tensorflow/models.git
    $ cd models
    $ protoc object_detection/protos/*.proto --python_out=.

Running

## Generating the TFRecord files
    $ wget <URL FOR TRAINING DATA>
    $ python create_tf_record.py  
    --data_dir <path to training data> --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_train.record --label_map_path  mbari_benthic_label_map.pbtxt --set train 
    $ python create_tf_record.py  
    --data_dir <path to training data> --collection MBARI_BENTHIC_2017 \
    --output_path MBARI_BENTHIC_2017_test.record --label_map_path  mbari_benthic_label_map.pbtxt --set test 
    
* Train model steps
* Test model steps

## Developer Notes

A placeholder for notes that might be useful for developers
 
* Install your own dataset (https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)
* Install TensorFlow Object Detection API (https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) 