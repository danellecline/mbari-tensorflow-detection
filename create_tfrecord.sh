#!/bin/bash
source $TF_VENV/bin/activate
python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017MSRCP \
--output_path MBARI_BENTHIC_2017MSRCP_train_subset.record \
--label_map_path $PWD/data/mbari_benthic_label_subset_map.pbtxt \
--set train \
--labels \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
SCOTOPLANES_GLOBOSA \

python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017MSRCP\
--output_path MBARI_BENTHIC_2017MSRCP_test_subset.record \
--label_map_path $PWD/data/mbari_benthic_label_subset_map.pbtxt \
--set test \
--labels \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
SCOTOPLANES_GLOBOSA \

exit
python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_train_peniagone_sp_a.record \
--label_map_path mbari_benthic_label_peniagone_sp_a.pbtxt \
--set train \
--labels \
PENIAGONE_SP_A \

python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_test_peniagone_sp_a.record \
--label_map_path mbari_benthic_label_peniagone_sp_a.pbtxt \
--set test \
--labels \
PENIAGONE_SP_A \
exit

python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_train.record \
--label_map_path mbari_benthic_label_map.pbtxt \
--set train \
--labels \
ELPIDIA \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
PENIAGONE_SP_1 \
PENIAGONE_PAPILLATA \
PENIAGONE_VITRAE \
SCOTOPLANES_GLOBOSA \

python create_tfrecord.py --data_dir \
$PWD/data \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_test.record \
--label_map_path mbari_benthic_label_map.pbtxt \
--set test \
--labels \
ELPIDIA \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
PENIAGONE_SP_1 \
PENIAGONE_PAPILLATA \
PENIAGONE_VITRAE \
SCOTOPLANES_GLOBOSA \
