from pylab import *
import glob
import os
import matplotlib.patches as mpatches
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))

import tensorflow as tf
from collections import namedtuple

model_metadata = namedtuple("model_metadata", ["meta_arch", "feature_extractor", "proposals", "dir", "name", "resolution"])
arch_markers = {'Faster RCNN': 'o', 'SSD':'D'}
fe_colors = {'Resnet 101':'Y', 'Inception V2':'B'}
sz_colors = {'950x540':'G', '300':'R', '600':'Y'}
arch_labels = []

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
  parser.add_argument('-m', '--model_dir', action='store', help='Root directory to raw dataset', required=False, default='{0}/models'.format(os.getcwd()))
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


def model_plot(all_model_index, model, ax):
  data = all_model_index.loc[model.name]
  m = '.'
  c = 'B'
  label = None
  if model.meta_arch in arch_markers.keys():
    m = arch_markers[model.meta_arch]
  if model.feature_extractor in fe_colors.keys():
    c = fe_colors[model.feature_extractor]
  if model.meta_arch not in arch_labels:
    label = model.meta_arch
    arch_labels.append(label)
  ax.scatter(data.index, data.values, marker=m, color=c, s=40, label=label)


def main(_):
  args = process_command_line()
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
  all_models = []

  for d in all_dirs:
    fc = 'Unknown'
    if 'resnet101' in d:
      fc = 'Resnet 101'
    if 'inception_v2' in d:
      fc = 'Inception v2'
    ma = 'Unknown'
    if 'ssd' in d:
      ma = 'SSD'
    if 'faster_rcnn' in d:
      ma = 'Faster RCNN'

    resolution = '960x540'
    proposals = 0
    dir_name = d.split('eval')[0]
    model_name = dir_name.split('/')[-2]
    f = model_name.split('_')
    for j in f:
      if j.isnumeric():
        proposals = int(j)
      #TODO add regex for resolution here

    # Grab all of the accuracy results for each model and put into Pandas dataframe
    event_acc = EventAccumulator(d)
    event_acc.Reload()
    # Show all tags in the log file
    print(event_acc.Tags())
    try:
      s = event_acc.Scalars('PASCAL/Precision/mAP@0.5IOU')
      df = pd.DataFrame(s)
      if df.empty:
        continue

      a = model_metadata(dir=d, name=model_name, meta_arch=ma, feature_extractor=fc, proposals=proposals, resolution=resolution)
      all_models.append(a)

      time_start = df.wall_time[0]

      # convert wall time and value to rounded values
      df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
      df['value'] = df['value'].apply(valueTomAP)

      # rename columns
      df.columns = ['GPU Time', 'step', 'Overall mAP']
      df['model'] = np.full(len(df), model_name)
      print(df)
      df_eval = df_eval.append(df)


    except Exception as ex:
      print(ex)
      continue

  # drop the step column as it's no longer needed
  df_eval = df_eval.drop(['step'], axis=1)
  # pivot on the same and plot the accuracy per each model
  #pivoted = df_eval.pivot(index=None, columns='model')

  #group = df_eval.groupby(['model'])
  all_model_index = df_eval.set_index(['model','GPU Time']).sort_index()

  with plt.style.context('ggplot'):

    # start a new figure - size is in inches
    #fig = plt.figure(figsize=(8, 10), dpi=400)
    fig = plt.figure(figsize=(8, 10))
    ax1 = plt.subplot(aspect='equal')
    ax1.set_xlim(0, 300)
    ax1.set_ylim(0, 100)

    for model in all_models:
      model_plot(all_model_index, model, ax1)

    #ax1.set_xlim(tmin, tmax)
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('mAP', fontsize=10)
    ax1.set_xlabel('GPU Time', fontsize=10)
    ax1.set_title('Foobar', fontstyle='italic')

    # plot the legend outside the plot in the upper left corner
    l = ax1.legend(loc='upper left', bbox_to_anchor=(0.5, 1), prop={'size': 8}, scatterpoints=1)
    l.set_zorder(4)  # put the legend on top right

    inc = 20
    '''ax1.text(170, 50, r'Resolution', fontsize=8)
    for feature, color in fe_colors.items():
      ax1.text(170, inc - 2, r'{0}'.format(feature), fontsize=8)
      c = mpatches.Circle( (160, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc += 10'''

    ax1.text(170, 50, r'Resolution', fontsize=8)
    for size, color in sz_colors.items():
      ax1.text(170, inc - 2, r'{0}'.format(size), fontsize=8)
      c = mpatches.Circle( (160, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc += 10

    #patches = [ mpatches.Patch(color=color, label=label)
    #  for label, color in zip(fe_labels, fe_colors)]
    #fig.legend(patches, fe_labels, loc='center', frameon=False)
    plt.show()
  #pivoted.plot(kind='bar', alpha=0.75, rot=45, figsize=(500, 500), width=.5)
  print('Done')

  #shutil.rmtree(eval_tempdir.name)
  #shutil.rmtree(train_tempdir.name)

if __name__ == '__main__':
  tf.app.run()
