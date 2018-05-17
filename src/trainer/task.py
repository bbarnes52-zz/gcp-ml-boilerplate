# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import posixpath
import shutil
import sys

import tensorflow as tf
from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

from constants import constants


def _parse_arguments(argv):
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Runs training on THD data.")
  parser.add_argument(
      "--model_dir", required=True,
      help="The directory where model outputs will be written")
  parser.add_argument(
      "--input_dir", required=True,
      help=("GCS or local directory containing tensorflow-transform outputs."))
  parser.add_argument(
      "--learning_rate", default=0.0005, required=False, type=float,
      help=("Learning rate to use during training."))
  parser.add_argument(
      "--batch_size", default=16, required=False, type=int,
      help=("Batch size to use during training."))
  parser.add_argument(
      "--num_epochs", default=20, required=False, type=int,
      help=("Number of epochs through the training set."))
  parser.add_argument(
      "--train_steps", default=100000, required=False, type=int,
      help=("Number of training steps."))
  args, _ = parser.parse_known_args(args=argv[1:])
  return args


def _extract_label(d):
  label = d.pop(constants.LABEL_COLUMN)
  return d, label


def _get_feature_columns():
  return [
    tf.feature_column.numeric_column(name)
    for name in constants.FEATURE_COLUMNS
  ]


def get_input_fn(file_pattern, feature_spec, num_epochs, batch_size):
  print file_pattern

  def _input_fn():
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(lambda x: tf.parse_single_example(x, feature_spec))
    dataset = dataset.map(lambda x: _extract_label(x))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

  return _input_fn


def get_serving_input_fn(input_dir):
  """Creates operations to ingest data for inference

    Args:
      input_dir: Directory containing tf.Transform metadata and transform_fn.
    Returns:
      A serving input function.
    """
  raw_metadata = metadata_io.read_metadata(
      posixpath.join(input_dir, constants.RAW_METADATA_DIR))
  transform_fn_path = posixpath.join(input_dir, constants.TRANSFORM_FN_DIR)
  return input_fn_maker.build_parsing_transforming_serving_input_fn(
      raw_metadata=raw_metadata,
      transform_savedmodel_dir=transform_fn_path,
      raw_label_keys=[constants.LABEL_COLUMN])


def run(args):
  #config = tf.estimator.RunConfig(save_checkpoints_steps=10)
  feature_spec = metadata_io.read_metadata(
      posixpath.join(args.input_dir, constants.TRANSFORMED_METADATA_DIR)).schema.as_feature_spec()
  train_input_fn = get_input_fn(
      "{}*".format(posixpath.join(args.input_dir, constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX)),
      feature_spec,
      num_epochs = args.num_epochs,
      batch_size = args.batch_size)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  print sess.run(train_input_fn())
  """
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=args.train_steps)
  eval_input_fn = get_input_fn(
      posixpath.join(args.input_dir, constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX),
      feature_spec,
      num_epochs = 1,
      batch_size = args.batch_size)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      exporters=tf.estimator.FinalExporter(
        name='export',
        serving_input_receiver_fn=get_serving_input_fn(args.input_dir)
      )
  )
  linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=_get_feature_columns(),
    model_dir=args.model_dir)
  tf.estimator.train_and_evaluate(linear_regressor, train_spec, eval_spec)
  """


if __name__ == '__main__':
    args = _parse_arguments(sys.argv)
    shutil.rmtree(args.model_dir, ignore_errors=True)
    run(args)
