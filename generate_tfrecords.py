"""
Generates tf.train.Example protos and writes them to TFRecord files
"""

import argparse
import os
import sys
import cv2
import numpy as np
import re
import scipy.io as sio
import tensorflow as tf

def open_img(img_name):
  img = cv2.imread(img_name)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # img = img.astype(np.float32)

  return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(input_file, output_file, data_dir, seq_length, num_keypoints):
  """Converts a file to TFRecords."""
  train_output_file = 'train_' + output_file
  eval_output_file = 'eval_' + output_file
  print('Generating %s and %s' % (train_output_file, eval_output_file))
  print('input_file %s' % input_file)
  with tf.python_io.TFRecordWriter(train_output_file) as train_writer, tf.python_io.TFRecordWriter(eval_output_file) as eval_writer:
    file_handle = open(input_file, 'r')
    for line in file_handle:
      line = line.strip()
      line = line.split(' ')
      img_name = os.path.join(data_dir,line[0])
      print('Image %s' % img_name)
      features = {}
      for j in range(seq_length):
        img = open_img(re.sub('_s00','_s'+str(j).zfill(2),img_name))
        joints = list(map(int,line[(j * (num_keypoints*2 + 1)) + 1:(j * (num_keypoints*2 + 1)) + num_keypoints*2 + 1]))
        
        features['image_' + str(j)] = _bytes_feature(img.tostring())
        features['joints_' + str(j)] = _int64_feature(joints)

      example = tf.train.Example(features=tf.train.Features(
          feature=features))

      # Do a 10% split for train and eval sets
      if np.random.choice([True, False],p=[0.9, 0.1]):
        train_writer.write(example.SerializeToString())
      else:
        eval_writer.write(example.SerializeToString())


def main(data_dir,input_file,seq_length,num_keypoints):
  
  output_file = input_file[:-3] + 'tfrecords'
  try:
    os.remove(output_file)
  except OSError:
    pass
    # Convert to tf.train.Example and write the to TFRecords.
  convert_to_tfrecord(input_file, output_file, data_dir, seq_length, num_keypoints)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory which contains images.')

  parser.add_argument(
      '--input-file',
      type=str,
      default='',
      help='File which contains image names and their corresponding lables.')
  parser.add_argument(
      '--seq-length',
      type=int,
      default=4,
      help='Sequence length.')
  parser.add_argument(
      '--num-keypoints',
      type=int,
      default=36,
      help='Total number of annotated keypoints.')
  args = parser.parse_args()
  main(args.data_dir, args.input_file, args.seq_length, args.num_keypoints)
