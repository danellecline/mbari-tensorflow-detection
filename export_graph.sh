#!/usr/bin/env bash
# Export graph of traing model 
# usage: export_graph.sh <model.ckpt> <output_directory>
# e.g. export_graph.sh faster_rcnn_resnet101_coco_20_smallanchor0 model.ckpt-1924
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/"$1"/pipeline_devbox.config \
    --trained_checkpoint_prefix models/"$2"/"$3" \
    --output_directory models/"$2"
