#!/usr/bin/env bash
# Export graph of traing model 
# usage: export_graph.sh <modelname> <model.ckpt> 
# e.g. export_graph.sh faster_rcnn_resnet101_coco_20_smallanchor model.ckpt-1924 
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
export CUDA_VISIBLE_DEVICES=0

OUT_DIR=models/export/"$1"
mkdir -p $OUT_DIR 
python tensorflow_models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/"$1"/pipeline_devbox.config \
    --trained_checkpoint_prefix models/"$1"/train/"$2" \
    --output_directory $OUT_DIR
