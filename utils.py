import os
import glob
import random

def ensure_dir(d):
  """
  ensures a directory exists; creates it if it does not
  :param fname:
  :return:
  """
  if not os.path.exists(d):
    os.makedirs(d)


def split(collection_dir, train_per, test_per):
  '''
  Split annotations into train/test in a particular collection_dir
  Outputs the split in two files named train.txt and test.txt
  :param collection_dir:
  :param train_per:
  :param test_per:
  :return:
  '''

  annotations = []
  for xml_in in glob.iglob(collection_dir + '/**/*.xml', recursive=True):
      print('Found {0}'.format(xml_in))
      path, filename = os.path.split(xml_in)
      annotations.append(filename)

  print('Found {0} xml annotations in {1}'.format(len(annotations), collection_dir))

  # split randomly
  random.shuffle(annotations)

  total = len(annotations)
  train_data = annotations[:round(train_per * total)]
  test_data = annotations[:round(test_per * total)]

  ensure_dir(collection_dir)
  with open(os.path.join(collection_dir, 'train.txt'), 'w') as f:
      for l in train_data:
          path, filename = os.path.split(l)
          f.write(filename + '\n')

  with open(os.path.join(collection_dir, 'test.txt'), 'w') as f:
      for l in test_data:
          path, filename = os.path.split(l)
          f.write(filename + '\n')