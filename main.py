#!/usr/bin/python3

import shutil
import tensorflow as tf
from data_pre import get_data
from model import Net
from data_set import data_set_dict

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('prefix', '/home/drc/Documents/UCR_TS_Archive_2015/',
                       'data set url prefix')
tf.flags.DEFINE_string('mode', 'cnn', 'cnn or dwt')
tf.flags.DEFINE_string('data_set', 'Adiac', 'data set name')
tf.flags.DEFINE_integer('slice_len', 0, 'data length after slice')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_bool('retrain', False, 'force to train or not')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_float('leakiness', 0.0, 'leakiness')
tf.flags.DEFINE_string('wavelet', 'db1', 'choose wavelet')
tf.flags.DEFINE_integer('steps', 1000, 'max training steps')
tf.flags.DEFINE_integer('max_level', 2, 'max dwt times')


def main(_):
    train_set = [
        FLAGS.prefix + FLAGS.data_set + '/' + FLAGS.data_set + '_TRAIN'
    ]
    eval_set = [FLAGS.prefix + FLAGS.data_set + '/' + FLAGS.data_set + '_TEST']
    model_url = './model/' + FLAGS.mode + '/' + FLAGS.data_set
    data_set = data_set_dict[FLAGS.data_set]

    if FLAGS.retrain:
        shutil.rmtree(model_url)

    model = Net(FLAGS.mode)

    hps = {
        'learning_rate': FLAGS.learning_rate,
        'leakiness': FLAGS.leakiness,
        'wavelet': FLAGS.wavelet,
        'max_level': FLAGS.max_level
    }
    print(hps)

    estimator = tf.estimator.Estimator(model.model_fn, model_url, params=hps)

    tensors_to_log = {'probabilities': 'softmax'}
    logging_hook = tf.train.LoggingTensorHook(tensors_to_log, 100, at_end=True)

    estimator.train(
        lambda: get_data(train_set, data_set.length, data_set.classes_num, True, FLAGS.slice_len, FLAGS.batch_size),  #pylint: disable=line-too-long
        [logging_hook],
        steps=FLAGS.steps)

    result = estimator.evaluate(
        lambda: get_data(eval_set, data_set.length, data_set.classes_num, False, FLAGS.slice_len, data_set.test_size),  #pylint: disable=line-too-long
        steps=1)

    result['error'] = 1 - result['accuracy']
    print(result)


if __name__ == '__main__':
    tf.app.run()
