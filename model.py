#!/usr/bin/python3

import tensorflow as tf
import numpy as np


class Net:
    def __init__(self, mode):
        self.__hps = None
        self.training = None
        self.__mode = mode
        return

    def model_fn(self, mode, features, labels, params):
        self.__hps = params
        self.training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        logits = self.__fn(features, labels.get_shape()[1])
        predictions = {
            'classes': tf.argmax(logits, 1),
            'probabilities': tf.nn.softmax(logits, name='softmax')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions)

        loss = tf.losses.softmax_cross_entropy(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            #  对于大量数据请修改优化器加速收敛
            optimizer = tf.train.GradientDescentOptimizer(
                self.__hps['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={
                'accuracy':
                tf.metrics.accuracy(
                    tf.argmax(labels, 1), predictions['classes'])
            })

#  残差组

    def __residual_stack(self, x, n, out_channels, stride, bottleneck):
        for i in range(n):
            if i != 0:
                stride = 1
            x = self.__residual_block(x, out_channels, stride, bottleneck)
        return x

#  残差单元

    def __residual_block(self, x, out_channels, stride, bottleneck):
        in_channels = x.get_shape()[-1]
        origin_x = x
        if bottleneck:
            x = tf.layers.conv1d(
                x, out_channels, 1, padding='same', strides=stride)
            x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.nn.leaky_relu(x, self.__hps['leakiness'])

            x = tf.layers.conv1d(x, out_channels, 3, padding='same')
            x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.nn.leaky_relu(x, self.__hps['leakiness'])

            x = tf.layers.conv1d(x, out_channels * 4, 1, padding='same')
            x = tf.layers.batch_normalization(x, training=self.training)
        else:
            x = tf.layers.conv1d(
                x, out_channels, 3, padding='same', strides=stride)
            x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.nn.leaky_relu(x, self.__hps['leakiness'])

            x = tf.layers.conv1d(x, out_channels, 3, padding='same')
            x = tf.layers.batch_normalization(x, training=self.training)

        out_channels = x.get_shape()[-1]
        if in_channels != out_channels or stride != 1:
            origin_x = tf.layers.conv1d(
                origin_x, out_channels, 1, padding='same', strides=stride)
            origin_x = tf.layers.batch_normalization(
                origin_x, training=self.training)

        return tf.nn.leaky_relu(x + origin_x, self.__hps['leakiness'])


#  离散小波变换

    @staticmethod
    def __dwt(x, wavelet):
        filters = wavelet.dec_lo[::-1]
        filters = np.array(filters, 'float32').reshape([wavelet.dec_len, 1, 1])
        channels = x.get_shape()[-1]
        splits = tf.split(x, channels, -1)
        for i in range(channels):
            splits[i] = tf.nn.conv1d(splits[i], filters, 2, 'SAME')
        return tf.concat(splits, -1)

    def __resnet(self, x):
        #  初始特征提取并激活
        x = tf.layers.conv1d(x, 64, 7, padding='same', strides=2)
        x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.leaky_relu(x, self.__hps['leakiness'])
        #  残差特征提取
        x = tf.layers.max_pooling1d(x, 3, 2, padding='same')
        x = self.__residual_stack(x, 3, 64, 1, True)

        x = self.__residual_stack(x, 4, 128, 2, True)

        x = self.__residual_stack(x, 6, 256, 2, True)

        x = self.__residual_stack(x, 3, 512, 2, True)
        #  全局平均，是否换成全连接？
        return tf.reduce_mean(x, 1)

        #  return tf.layers.dense(x, classes_num)

    def __fn(self, x, classes_num):
        x_list = [x]
        if self.__mode == 'dwt':
            for _ in range(self.__hps['dwt_times']):
                x = Net.__dwt(x, self.__hps['wavelet'])
                x_list.append(x)
        y_list = []
        for o in x_list:
            y_list.append(self.__resnet(o))
        y = tf.concat(y_list, 1)
        return tf.layers.dense(y, classes_num)
