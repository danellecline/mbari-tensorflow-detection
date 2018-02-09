#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''

Combine all GPU/CPU inference times into a single plot
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import matplotlib
matplotlib.use('Agg')
from pylab import *
import os
import pandas as pd
import glob
import model_metadata as meta
import matplotlib.pyplot as plt
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

if __name__ == '__main__':

    compute = ['GPU'] #['CPU', 'GPU']
    custom_colors = list(['b'])

    for c in compute:
      base_dir = os.path.join(os.getcwd(), 'data')
      all_files = sorted(glob.iglob(base_dir + '/**/*{0}*.csv'.format(c), recursive=True))
      df = pd.DataFrame()
      for f in all_files:
        data = pd.read_csv(f,index_col=None)
        d = os.path.basename(f)
        model_name = d.split('-')[1]
        m = meta.ModelMetadata(model_name)
        if data.empty:
          continue
        if m.proposals > 0:
            model_description = '{0}\n{1} Box Proposals'.format(m.meta_arch, m.proposals)
        elif m.image_resolution > 0 and m.meta_arch == 'SSD':
            model_description = '{0}\n{1} Image Resolution'.format(m.meta_arch, m.image_resolution)
        else:
          model_description = m.meta_arch
        df = df.append({'Model':model_description , '{0} Time'.format(c):int(data.iloc[0]['GPU Time'])}, ignore_index=True)

      # start a new figure - size is in inches
      fig = plt.figure(figsize=(8, 8), dpi=200)
      ax = plt.subplot()
      df = df.set_index('Model')
      df.plot(kind='bar', alpha=0.75, rot=30, legend=False, ax=ax, color=custom_colors)
      ax.set_xlabel("")
      ax.set_ylabel("Milliseconds", fontsize=10)
      ax.set_title(label='Average {0} Detection Time'.format(c), fontstyle='italic')
      plt.show()
      fig.savefig(fname='{0}_time_plot.png'.format(c))
      print('Done')
