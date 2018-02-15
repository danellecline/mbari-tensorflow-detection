#!/usr/bin/env bash
BASE_DIR=~/Dropbox/GitHub/mbari-tensorflow-detection 
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
python box_proposal_plot.py
scp mAP.png  dcline@mbari1913:'~/Dropbox/poster/mAP_MBARI_BENTHIC_ALL.png' 
