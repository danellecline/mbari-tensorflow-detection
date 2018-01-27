# basic constants
CLASSES = ['BENTHOCODON', 'ECHINOCREPIS', 'PENIAGONE_SP_1', 'PENIAGONE_SP_2',
           'PENIAGONE_SP_A', 'PENIAGONE_VITRAE', 'SCOTOPLANES_GLOBOSA', 'ELPIDIA']
 
ALL_CLASSES = ['ELPIDIA', 'PENIAGONE_PAPILLATA', 'ECHINOCREPIS', 'SCOTOPLANES_GLOBOSA',
               'ONEIROPHANTA_MUTABILIS_COMPLEX', 'PENIAGONE_SP_1', 'LONG_WHITE',
               'SYNALLACTIDAE', 'PENIAGONE_VITRAE', 'FUNGIACYATHUS_MARENZELLERI',
               'CYSTECHINUS_LOVENI', 'PENIAGONE_SP_A', 'BENTHOCODON', 'TJALFIELLA',
               'PENIAGONE_SP_2', 'FISH']
  
# directory where raw annotations are
BASE_DIR_RAW = '/Volumes/DeepLearningTests/nyee_datasets/frcnn_data/' 

# name of the data collection
COLLECTION_NAME = 'MBARI_BENTHIC_2017_300'

# base directory structure to store converted annotations
BASE_DIR_CONVERT = '/Volumes/DeepLearningTests/training_data_to_export/'

TRAIN_VID_KEYS = ['D008_03HD', 'D0232_04HD', 'D0442_06HD', 'D0443_05HD',  'D0673_04HD', 'D0772_09HD', 'D0904_D3HD']
TEST_VID_KEYS = ['D0232_03HD', 'D0772_10HD']

# directories for output 
PNG_DIR = 'PNGImages'
ANNOTATION_DIR = 'Annotations'

TARGET_WIDTH = 300
TARGET_HEIGHT = 300