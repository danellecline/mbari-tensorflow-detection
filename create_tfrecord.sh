#!/usr/bin/env bash


python create_tfrecord.py --data_dir \
/Volumes/DeepLearningTests/training_data_to_export/ \
--collection  MBARI_BENTHIC_2017_300 \
--output_path MBARI_BENTHIC_2017_300_train_subset.record \
--label_map_path mbari_benthic_label_subset_map.pbtxt \
--set train \
--labels \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
SCOTOPLANES_GLOBOSA \

python create_tfrecord.py --data_dir \
/Volumes/DeepLearningTests/training_data_to_export/ \
--collection  MBARI_BENTHIC_2017_300 \
--output_path MBARI_BENTHIC_2017_300_test_subset.record \
--label_map_path mbari_benthic_label_subset_map.pbtxt \
--set test \
--labels \
BENTHOCODON \
ECHINOCREPIS \
PENIAGONE_SP_A \
SCOTOPLANES_GLOBOSA \

exit
python create_tfrecord.py --data_dir \
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_train_peniagone_sp_a.record \
--label_map_path mbari_benthic_label_peniagone_sp_a.pbtxt \
--set train \
--labels \
PENIAGONE_SP_A \

python create_tfrecord.py --data_dir \
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
--collection  MBARI_BENTHIC_2017 \
--output_path MBARI_BENTHIC_2017_test_peniagone_sp_a.record \
--label_map_path mbari_benthic_label_peniagone_sp_a.pbtxt \
--set test \
--labels \
PENIAGONE_SP_A \
exit

python create_tfrecord.py --data_dir \
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
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
/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/ \
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
