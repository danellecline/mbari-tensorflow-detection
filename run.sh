#!/bin/bash
# Training command for executing on a DevBox
# Execute training with run_train_devbox.sh <model dir name> <test|train> <gpu device #>
# e.g. run.sh faster_rcnn_resnet50_coco test 0
# This will train model faster_rcnn_resnet50_coco on GPU device 0
# best to split the train/test for each model on different GPUs which each have 12GB of memory
# The models use a lot of memory during training, and can spike during
# testing at checkpoints
source $TF_VENV/bin/activate
pushd $PWD/tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/ 
export CUDA_VISIBLE_DEVICES="$3" 
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [ "$2" == "train" ]; then
echo "Training model $1 on GPU $3"
python $BASE_DIR/tensorflow_models/research/object_detection/train.py \
--logtostderr \
--pipeline_config_path=`pwd`/models/$1/pipeline_devbox.config \
--train_dir=`pwd`/models/$1/train/

elif [ "$2" == "test" ]; then
echo "Testing model $1 on GPU $3"
python $BASE_DIR/tensorflow_models/research/object_detection/eval.py \
--logtostderr \
--pipeline_config_path=`pwd`/models/$1/pipeline_devbox.config \
--checkpoint_dir=`pwd`/models/$1/train/ \
--eval_dir=`pwd`/models/$1/eval/ 
else
echo "$0: $2 is not a valid option. Choose test or train"
fi
