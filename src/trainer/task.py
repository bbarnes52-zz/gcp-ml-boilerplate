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

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
import model
import input_fn_utils
import argparse
import shutil
import sys


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
      "--num_epochs", default=5, required=False, type=int,
      help=("Number of epochs through the training set"))
  args, _ = parser.parse_known_args(args=argv[1:])
  return args


def _extract_label(d):
  label = d.pop(constants.LABEL_COLUMN)
  return d, label


def _get_feature_columns():
  return [
    tf.feature_column.numeric_column(name)
    for name in constants.FEATURE_COLUMN_NAMES
  ]


def read_dataset(file_pattern, feature_spec, num_epochs, batch_size):
  files = tf.data.Dataset.list_files(file_pattern)
  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.shuffle(1000)
  dataset = dataset.repeat(num_epochs)
  transformed_metadata = metadata_io.read_metadata(
      posix.path.join(input_dir, constants.TRANSFORMED_METADATA_DIR))
  dataset = dataset.map(lambda x: tf.parse_single_example(x, transformed_metadata.schema.as_feature_spec()))
  dataset = dataset.map(lambda x: _extract_label(x))
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()
  return features


def run(args):
  config = tf.contrib.learn.RunConfig(save_checkpoints_steps=1000)
  train_input_fn = read_dataset(
      posix.path.join(args.input_dir, constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX),
      num_epochs = args.num_epochs,
      batch_size = args.batch_size)
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=args.train_steps)
  eval_input_fn = input_pipeline.read_dataset(
      posix.path.join(args.input_dir, constants.TRANSFORMED_EVAL_DATA_FILE_PREFIX),
      num_epochs = 1,
      batch_size = args.batch_size)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      exporters=tf.estimator.FinalExporter(
        name='export',
        serving_input_fn=input_fn_utils.get_serving_input_fn(args.input_dir)
      )
  )
  linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=_get_feature_columns(),
    model_dir=args.model_dir,
    config=config)
  tf.estimator.train_and_evaluate(linear_regressor, train_spec, eval_spec)


if __name__ == '__main__':
    args = _parse_arguments(sys.argv)
    shutil.rmtree(args.model_dir, ignore_errors=True)
    run(args)
