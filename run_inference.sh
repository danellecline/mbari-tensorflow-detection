#!/usr/bin/env bash
# Run inference command on a DevBox
# Execute inference with run_inference.sh <tfrecord> <model> <gpu device>
# e.g. run_inference.sh MBARI_BENTHIC_2017_small.record faster_rcnn_resnet101_coco_300_smallanchor 0
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
pushd tensorflow_models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection
popd
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/
if [ "$3" == -1 ]
then
export CUDA_VISIBLE_DEVICES=""
CSV_FILE=`pwd`/data/"$1"-"$2"-CPU-time.csv
else
export CUDA_VISIBLE_DEVICES="$3"
CSV_FILE=`pwd`/data/"$1"-"$2"-GPU-time.csv
fi

TF_RECORD_FILE_OUT=`pwd`/data/"$1"-"$2"_detections_images.record
TF_RECORD_FILE_IN=`pwd`/data/"$1"
MODEL_DIR=`pwd`/models/export/"$2"/frozen_inference_graph.pb

python tensorflow_models/research/object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILE_IN \
  --output_tfrecord_path=$TF_RECORD_FILE_OUT \
  --csv_path=$CSV_FILE \
  --inference_graph=$MODEL_DIR 
#\
#  --discard_image_pixels
