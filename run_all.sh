#!/bin/bash
set -x
function tmux_start_train { tmux new-window -d  -n "train:GPU$1" "${*:2}; exec bash"; }
function tmux_start_test { tmux new-window -d  -n "test:GPU$1" "${*:2}; exec bash"; }

models=( ["0"]="ssd_inception_v2_coco_600" ["1"]="ssd_inception_v2_coco_300")
tmux new-session -d -s "train"
GPU_IDX=2
for gpu_index in $(seq 0 $(($GPU_IDX-1))); do
  tmux_start_train ${gpu_index} \
  ./run.sh ${models[gpu_index]} train ${gpu_index}
done
GPU_IDX=4
models=( ["2"]="ssd_inception_v2_coco_600" ["3"]="ssd_inception_v2_coco_300")
tmux new-session -d -s "test"
for gpu_index in $(seq 2 $(($GPU_IDX-1))); do
  tmux_start_test ${gpu_index} \
  ./run.sh ${models[gpu_index]} test ${gpu_index}
done
sleep 11h
tmux kill-session -t train
tmux kill-session -t test
sleep 5s
exit

# Run models for 6 hours each, splitting testing and training across 4 GPUS
GPU_IDX=2
models=( ["0"]="faster_rcnn_resnet101_coco_300" ["1"]="faster_rcnn_resnet101_coco_100")
tmux new-session -d -s "train"
for gpu_index in $(seq 0 $(($GPU_IDX-1))); do
  tmux_start_train ${gpu_index} \
  ./run.sh ${models[gpu_index]} train ${gpu_index} 
done
GPU_IDX=4
models=( ["2"]="faster_rcnn_resnet101_coco_300" ["3"]="faster_rcnn_resnet101_coco_100")
tmux new-session -d -s "test"
for gpu_index in $(seq 2 $(($GPU_IDX-1))); do
  tmux_start_test ${gpu_index} \
  ./run.sh ${models[gpu_index]} test ${gpu_index} 
done
sleep 6h
tmux kill-session -t train
tmux kill-session -t test

# allow few seconds for everything to die
