#!/usr/bin/env bash
# Training command for executing on our DevBox
# Execute training with run_train_devbox.sh <model dir name> <test|train>, e.g. run_train_devbox.sh faster_rcnn_resnet50_coco test
pushd tensorflow_models
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/

if $2 == 'train'; then

python tensorflow_models/object_detection/train.py \
--logtostderr \
--pipeline_config_path=`pwd`/models/$1/pipeline_devbox.config \
--train_dir=`pwd`/models/$1/train/ \
--eval_dir=`pwd`/models/$1/eval/

elif $2 == 'test'; then

python tensorflow_models/object_detection/eval.py \
--logtostderr \
--pipeline_config_path=`pwd`/models/$1/pipeline_devbox.config \
--checkpoint_dir=`pwd`/models/$1/train/ \
--eval_dir=`pwd`/models/$1/eval/

else
echo "$0: $2 is not a valid option. Choose test or train"
fi
