#!/usr/bin/env bash
# Run inference command on a DevBox
# Execute inference with run_inference.sh <tfrecord> <gpu device #>
# e.g. run_inference.sh MBARI_BENTHIC_2017_test.record 0
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
export CUDA_VISIBLE_DEVICES="$2"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

TF_RECORD_FILE="data/$1"

python -m object_detection/inference/infer_detections \
  --input_tfrecord_paths=$TF_RECORD_FILE \
  --output_tfrecord_path=test_detections.tfrecord-00000-of-00001 \
  --inference_graph=models/faster_rcnn_resnet101_coco_20_smallanchor0/frozen_inference_graph.pb 
  
#--discard_image_pixels
