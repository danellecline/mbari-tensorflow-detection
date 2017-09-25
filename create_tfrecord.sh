#!/usr/bin/env bash
python create_tfrecord.py --data_dir \
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_train.record \
--label_map_path mbari_benthic_label_map.pbtxt \
--set train \
--labels \
ELPIDIA \
BENTHOCODON \
CYSTECHINUS_LOVENI \
ECHINOCREPIS \
ELPIDIA \
FISH \
FUNGIACYATHUS_MARENZELLERI \
LONG_WHITE \
ONEIROPHANTA_MUTABILIS_COMPLEX \
PENIAGONE_SP_A \
PENIAGONE_SP_1 \
PENIAGONE_SP_2 \
PENIAGONE_PAPILLATA \
PENIAGONE_VITRAE \
SCOTOPLANES_GLOBOSA \
SYNALLACTIDAE \
TJALFIELLA \

python create_tfrecord.py --data_dir \
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_test.record \
--label_map_path mbari_benthic_label_map.pbtxt \
--set test \
--labels \
ELPIDIA \
BENTHOCODON \
CYSTECHINUS_LOVENI \
ECHINOCREPIS \
ELPIDIA \
FISH \
FUNGIACYATHUS_MARENZELLERI \
LONG_WHITE \
ONEIROPHANTA_MUTABILIS_COMPLEX \
PENIAGONE_SP_A \
PENIAGONE_SP_1 \
PENIAGONE_SP_2 \
PENIAGONE_PAPILLATA \
PENIAGONE_VITRAE \
SCOTOPLANES_GLOBOSA \
SYNALLACTIDAE \
TJALFIELLA \