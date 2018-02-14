#!/usr/bin/python3

import tensorflow as tf
from data_pre import get_data
from model import Net

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    train_set = [
        '/home/drc/Documents/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices_TRAIN'
    ]

    eval_set = [
        '/home/drc/Documents/UCR_TS_Archive_2015/ElectricDevices/ElectricDevices_TEST'
    ]

    model = Net()

    hps = {'learning_rate': 0.001, 'leakiness': 0.0}
    estimator = tf.estimator.Estimator(model.model_fn, './model', params=hps)

    tensors_to_log = {'probabilities': 'softmax'}
    logging_hook = tf.train.LoggingTensorHook(tensors_to_log, 100, at_end=True)

    estimator.train(
        lambda: get_data(train_set, 96, 7, True, batch_size=256),
        [logging_hook],
        max_steps=20000)

    result = estimator.evaluate(
        lambda: get_data(eval_set, 96, 7, False, batch_size=7711), steps=1)

    print(result)


if __name__ == '__main__':
    tf.app.run()
