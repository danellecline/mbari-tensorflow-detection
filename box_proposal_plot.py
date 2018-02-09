#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Combine all model data into a single plot
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''
from pylab import *
import glob
import os
import matplotlib.patches as mpatches
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_metadata as meta
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
plt.style.use('ggplot')
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
  parser.add_argument('-t', '--title', action='store', help='Title for plot', required=False, default='Foobar')
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

      dir_name = d.split('eval')[0]
      model_name = dir_name.split('/')[-2]

      a = meta.ModelMetadata(model_name)
      all_models.append(a)

      time_start = df.wall_time[0]

      # convert wall time and value to rounded values
      df['wall_time'] = df['wall_time'].apply(wallToGPUTime, args=(time_start,))
      df['value'] = df['value'].apply(valueTomAP)

      # rename columns
      df.columns = ['GPU Time', 'step', 'Overall mAP']
      df['model'] = np.full(len(df), a.name)
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
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax1 = plt.subplot(aspect='equal')
    ax1.set_xlim(0, 300)
    ax1.set_ylim(0, 100)

    for model in all_models:
      model_plot(all_model_index, model, ax1)

    ax1.set_ylim([0, 100])
    ax1.set_ylabel('mAP', fontsize=10)
    ax1.set_xlabel('Box Proposals', fontsize=10)
    ax1.set_title(args.title, fontstyle='italic')

    # plot the legend outside the plot in the upper left corner
    l = ax1.legend(loc='upper left', bbox_to_anchor=(0.5, 0.95), prop={'size': 8}, scatterpoints=1, title='Architecture')
    l.get_title().set_fontsize('8')
    l.set_zorder(4)  # put the legend on top right
    inc = 20
    '''ax1.text(170, 50, r'Resolution', fontsize=8)
    for feature, color in fe_colors.items():
      ax1.text(170, inc - 2, r'{0}'.format(feature), fo ntsize=8)
      c = mpatches.Circle( (160, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc += 10'''

    ax1.text(160, 50, r'Resolution', fontsize=8)
    for size, color in sz_colors.items():
      ax1.text(170, inc - 2, r'{0}'.format(size), fontsize=8)
      c = mpatches.Circle( (160, inc), 2, edgecolor='black', facecolor=color)
      ax1.add_patch(c)
      inc += 10

    #patches = [ mpatches.Patch(color=color, label=label)
    #  for label, color in zip(fe_labels, fe_colors)]
    #fig.legend(patches, fe_labels, loc='center', frameon=False)
    plt.savefig('{0}.png'.format(args.title), format='png')
    plt.show()
  #pivoted.plot(kind='bar', alpha=0.75, rot=45, figsize=(500, 500), width=.5)
  print('Done')

  #shutil.rmtree(eval_tempdir.name)
  #shutil.rmtree(train_tempdir.name)

if __name__ == '__main__':
  tf.app.run()