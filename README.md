# mbari-tensorflow-detection
 
Code for testing Tensorflow detection API on underwater imagery

## Prerequisites
 
- Python version  3.5.4, on a Mac OSX download and install from here:
 https://www.python.org/downloads/mac-osx/ 

## Running

Create virtual environment with correct dependencies

    $ pip3 install virtualenv
    $ virtualenv --python=/usr/local/bin/python3.5 venv-pam
    $ source venv-pam/bin/activate
    $ pip3 install -r requirements.txt
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
    $ pip3 install --upgrade $TF_BINARY_URL

Check-out code

    $ git clone https://github.com/danellecline/mbari-tensorflow-detection

TODO: Add steps for running below

* Convert annotations to tfrecords
* Train model steps
* Test model steps

## Developer Notes

A placeholder for notes that might be useful for developers

* Install Tensorflow from source https://www.tensorflow.org/install/install_sources
* Install your own dataset https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md
 