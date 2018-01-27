import hashlib
import conf
import glob
import io
import os
import logging
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from collections import namedtuple

models = ['faster_rcnn_resnet101_coco_100_smallanchor', 'faster_rcnn_resnet101_coco_300_smallanchor']
markers = {'faster_rcnn': 'o', 'ssd':'D'}
colors = {'faster_rcnn':'Y', 'ssd':''}

def process_command_line():
  '''
  Process command line
  :return: args object
  '''

  import argparse
  from argparse import RawTextHelpFormatter

  examples = 'Examples:' + '\n\n'
  examples += 'Extract and plot performance metrics from model output \n'
  examples += '{0} --model_dir {0}/models'.format(os.getcwd())
  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                   description='Creates Tensorflow Record object for MBARI annotated data',
                                   epilog=examples)
  parser.add_argument('-m', '--model_dir', action='store', help='Root directory to raw dataset', required=False, default='{0}/models/test'.format(os.getcwd()))
  args = parser.parse_args()
  return args

def aggregate(search_path, tempdir):
  all_files = glob.glob(search_path, recursive=True)
  for f in all_files:
    src = f
    dir, file = os.path.split(src)
    dst = '{0}/{1}'.format(tempdir, file)
    shutil.copy(src, dst)

def wallToGPUTime(x, zero_time):
  return round(int((x - zero_time)/60),0)

def valueTomAP(x):
  return round(int(x*100),0)

def modelToMetaArch(x):
  if 'faster_rcnn' in x:
    return 'faster_rcnn'
  return 'Unknown'


def model_plot(all_model_index, model_name):
  data = all_model_index.loc[model_name]
  m = '.'
  c = 'B'
  if model_name in markers.keys():
    m = markers[model_name]
  if model_name in colors.keys():
    c = colors[model_name]
  plt.scatter(data.index, data.values, marker=m, color=c, s=40)

  #ax.set_xlim(tmin, tmax)
  #ax.set_ylim([dmax, dmin])
  #ax.set_ylabel('depth (m)', fontsize=8)

  #ax.tick_params(axis='both', which='major', labelsize=8)
  #ax.tick_params(axis='both', which='minor', labelsize=8)
  #cs = ax.scatter(x, y, c=z, s=20, marker='.', vmin=zmin, vmax=zmax, lw=0, alpha=1.0, cmap=self.cm_jetplus)


def main(_):
  args = process_command_line()

  model_name_map = {}
  model_name_map['faster_rcnn_inception_resnet_v2_atrous_coco_100'] = ''
  import tempfile
  import shutil

  output = os.getcwd()
  #train_tempdir = tempfile.TemporaryDirectory()
  #eval_tempdir = tempfile.TemporaryDirectory()

  #aggregate(args.model_dir + '/**/train/events*', train_tempdir.name)
  #aggregate(args.model_dir + '/**/eval/events*', eval_tempdir.name)

  search_path = args.model_dir + '/**/eval/'
  all_dirs = glob.glob(search_path, recursive=True)
  df_eval = pd.DataFrame()

  # Grab all of the accuracy results for each model and put into Pandas dataframe
  for d in all_dirs:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(d)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    try:
      s = event_acc.Scalars('PASCAL/Precision/mAP@0.5IOU')
      df = pd.DataFrame(s)
      time_start = df.wall_time[0]
      df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
      df['value'] = df['value'].apply(valueTomAP)
      df['Meta Architecture'] = df['value']

      # rename columns
      df.columns = ['GPU Time', 'step', 'Overall mAP']
      s = d.split('models')
      dir_name = s[1].split('eval')[0]
      model_name = dir_name.split('/')[-2]

      # remap the model name to better name for the plot
      #model_name = model_name_map[dir_name]
      df['model'] = np.full(len(df), model_name)
      print(df)
      df_eval = df_eval.append(df)
    except Exception as ex:
      print(ex)
      continue

  #drop the step column as it's no longer needed
  df_eval = df_eval.drop(['step'], axis=1)

  # pivot on the same and plot the accuracy per each model
  #pivoted = df_eval.pivot(index=None, columns='model')

  #group = df_eval.groupby(['model'])
  all_model_index = df_eval.set_index(['model','GPU Time']).sort_index()

  # start a new figure - size is in inches
  #fig = plt.figure(figsize=(8, 10))
  #fig.suptitle(self.title + '\n' + self.subtitle1 + '\n' + self.subtitle2, fontsize=8)

  for model_name in models:
    model_plot(all_model_index, model_name)

  plt.legend(models)

  #data = all_names_index.loc[sex, name]
 # plt.plot(pivoted)
  plt.show()
  print('Done')

  #pivoted.plot(kind='bar', alpha=0.75, rot=45, figsize=(500, 500), width=.5)
  print('Done')


    #shutil.rmtree(eval_tempdir.name)
  #shutil.rmtree(train_tempdir.name)

if __name__ == '__main__':
  tf.app.run()
