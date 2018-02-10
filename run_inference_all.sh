#!/bin/bash
set -x
function tmux_start_inference { tmux new-window -d  -n "inference:GPU$1" "${*:2}; exec bash"; }

models=(
["0"]="faster_rcnn_resnet101_coco_300_smallanchor" \
["1"]="faster_rcnn_resnet101_coco_100_smallanchor" \
["2"]="faster_rcnneresnet101_coco_50_smallanchor" \
["3"]="rfcn_resnet101_coco_300_smallanchor" \
["4"]="rfcn_resnet101_coco_100_smallanchor" \
["5"]="rfcn_resnet101_coco_50_smallanchor" \
["6"]="ssd_inception_v2_coco_600" \
["7"]="ssd_inception_v2_coco_300" \
)
NUM_MODELS=8

# Run inference across GPU last, using four models at a time
for i in $(seq 0 4 $(($NUM_MODELS-1))); do
  tmux new-session -d -s "inference"
  tmux_start_inference 0 ./run_inference.sh MBARI_BENTHIC_2017_small.record ${models[i]} 0
  tmux_start_inference 1 ./run_inference.sh MBARI_BENTHIC_2017_small.record ${models[i + 1]} 1
  tmux_start_inference 2 ./run_inference.sh MBARI_BENTHIC_2017_small.record ${models[i + 2]} 2
  tmux_start_inference 3 ./run_inference.sh MBARI_BENTHIC_2017_small.record ${models[i + 3]} 3
  sleep 2m
  tmux kill-session -t inference
done
exit

# Run inference each across CPU  last
for i in $(seq 0 1 $(($NUM_MODELS-1))); do
  ./run_inference.sh MBARI_BENTHIC_2017_small.record ${models[i]} -1
done
