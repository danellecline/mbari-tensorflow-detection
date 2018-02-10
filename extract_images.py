import matplotlib
matplotlib.use('Agg')
import hashlib
import conf
import io
import os
import logging
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_models/research'))
import tensorflow as tf
import cv2
from object_detection.utils import visualization_utils
from object_detection.utils import label_map_util

def process_command_line():
  '''
  Process command line
  :return: args object
  '''

  import argparse
  from argparse import RawTextHelpFormatter

  examples = 'Examples:' + '\n\n'
  examples += 'Extract images from a tensorflow record :\n'
  examples += '{0} --record MBARI_BENTHIC_2017_test.record'.format(sys.argv[0])
    
  parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                   description='Extract images from a tensorflow record ',
                                   epilog=examples)
  parser.add_argument('-r', '--record', action='store', help='Name of the collection. Also the subdirectory name '
                                                                 'for the raw dataset', required=True)
  parser.add_argument('-o', '--output_path', action='store', help='Path to save images to', required=True)
  parser.add_argument('-l', '--label_map_path',action='store', help='Path to label map proto', required=True)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = process_command_line()
  tfrecords_filename = args.record
  output_path = args.output_path
  c = 0
  category_index = label_map_util.create_category_index_from_labelmap(args.label_map_path)

  for record in tf.python_io.tf_record_iterator(tfrecords_filename):
      c += 1

  tf.reset_default_graph()

  fq = tf.train.string_input_producer([tfrecords_filename], num_epochs=c)
  reader = tf.TFRecordReader()
  _, v = reader.read(fq)
  fk = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
      'image/detection/label': tf.VarLenFeature(tf.int64),
      'image/detection/score': tf.VarLenFeature(tf.float32),
      'image/detection/bbox/ymin': tf.VarLenFeature(tf.float32),
      'image/detection/bbox/ymax': tf.VarLenFeature(tf.float32),
      'image/detection/bbox/xmin': tf.VarLenFeature(tf.float32),
      'image/detection/bbox/xmax': tf.VarLenFeature(tf.float32)
      }

  ex = tf.parse_single_example(v, fk)
  image = tf.image.decode_jpeg(ex['image/encoded'], dct_method='INTEGER_ACCURATE')
  fname = tf.cast(ex['image/filename'], tf.string)
  labels = tf.cast(ex['image/detection/label'], tf.int64)
  scores = tf.cast(ex['image/detection/score'], tf.float32)
  ymin = tf.cast(ex['image/detection/bbox/ymin'], tf.float32)
  ymax = tf.cast(ex['image/detection/bbox/ymax'], tf.float32)
  xmin = tf.cast(ex['image/detection/bbox/xmin'], tf.float32)
  xmax = tf.cast(ex['image/detection/bbox/xmax'], tf.float32)

  # The op for initializing the variables.
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  with tf.Session()  as sess:
      sess.run(init_op)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      # sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

      # set the number of images in your tfrecords file
      num_images=c
      print("going to restore {} files from {}".format(num_images, tfrecords_filename))
      for i in range(num_images):

          im_,labels_,scores_,fname_,xmin_,xmax_,ymin_,ymax_ = sess.run([image,labels,scores,fname,xmin,xmax,ymin,ymax])
          boxes = np.vstack((ymin_.values,xmin_.values,ymax_.values,xmax_.values)).transpose()
          fname_f = fname_.decode("utf-8")
          print('{0} '.format(fname_f))
          im_ = visualization_utils.visualize_boxes_and_labels_on_image_array(im_,
                                          boxes,
                                          labels_.values,
                                          scores_.values,
                                          category_index,
                                          instance_masks=None,
                                          keypoints=None,
                                          use_normalized_coordinates=True,
                                          max_boxes_to_draw=20,
                                          min_score_thresh=.5,
                                          agnostic_mode=False,
                                          line_thickness=4)

          if not os.path.exists(output_path):
              os.makedirs(output_path)
          fName_=os.path.join(output_path, fname_f)
          cv2.imwrite(fName_ , im_)

      coord.request_stop()
      coord.join(threads)
