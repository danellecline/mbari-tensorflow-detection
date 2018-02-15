#!/usr/bin/env bash
BASE_DIR=~/Dropbox/GitHub/mbari-tensorflow-detection 
source ~/Dropbox/GitHub/venv-aesa-tensorflow-detection-devbox/bin/activate
python accuracy_by_species.py
scp mAPPENIAGONE_SP_A.png  dcline@mbari1913:~/Dropbox/poster/ 
scp mAPECHINOCREPIS.png  dcline@mbari1913:~/Dropbox/poster/
scp mAPSCOTOPLANES_GLOBOSA.png dcline@mbari1913:~/Dropbox/poster/
scp mAPBENTHOCODON.png dcline@mbari1913:~/Dropbox/poster/ 
