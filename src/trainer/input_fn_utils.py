# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import os
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.saved import input_fn_maker

from constants import constants


TARGET_FEATURE_COLUMN='shelf_out'


def read_and_decode_single_record(input_dir, num_epochs,
        mode=tf.contrib.learn.ModeKeys.TRAIN):
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
      num_epochs = num_epochs
      file_prefix = constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX
  else:
      num_epochs = 1
      # TODO(bgb): Use proper file prefix once eval set is available.
      file_prefix = constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX
      #file_prefix = constants.TRANSFORMED_EVAL_DATA_FILE_PREFIX

  transformed_metadata = metadata_io.read_metadata(os.path.join(input_dir,
      constants.TRANSFORMED_METADATA_DIR))
  input_file_names = tf.train.match_filenames_once(os.path.join(input_dir,
      '{}*{}'.format(file_prefix, constants.DATA_FILE_SUFFIX)))
  filename_queue = tf.train.string_input_producer(input_file_names,
          num_epochs=num_epochs, shuffle=True)

  # TODO(bgb): Add conditional to check for file extension type.
  reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP))
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized = serialized_example,
      features=transformed_metadata.schema.as_feature_spec()
  )
  return features


def read_dataset(input_dir, num_epochs, batch_size, mode=tf.contrib.learn.ModeKeys.TRAIN):
  def _input_fn():
    min_after_dequeue = 10000
    features = read_and_decode_single_record(input_dir, num_epochs, mode)
    features = tf.train.shuffle_batch(
            tensors=features,
            batch_size=batch_size,
            min_after_dequeue=min_after_dequeue,
            capacity=(min_after_dequeue + 3) * batch_size)
    target = features.pop(TARGET_FEATURE_COLUMN)
    return features, target
  return _input_fn


def get_serving_input_fn(input_dir):
  raw_metadata = metadata_io.read_metadata(os.path.join(input_dir,
      constants.RAW_METADATA_DIR))
  transform_fn_path = os.path.join(input_dir, constants.TRANSFORM_FN_DIR)
  return input_fn_maker.build_parsing_transforming_serving_input_fn(
          raw_metadata=raw_metadata,
          transform_savedmodel_dir=transform_fn_path,
          raw_label_keys=[TARGET_FEATURE_COLUMN])
