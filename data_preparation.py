#!/usr/bin/python
import numpy as np


def _load_data(path):
    data = np.loadtxt(path, delimiter=',')
    rng = np.random.RandomState(472943)
    idn = np.arange(data.shape[0])
    rng.shuffle(idn)
    data = data[idn]
    return data


def _slice_data(data, length):
    ori_num = data.shape[0]
    ori_len = data.shape[1]
    threshold = min(ori_len - 1, length) / 2
    length = 1
    while length <= threshold:
        length <<= 1
    length += 1
    mul = ori_len - length + 1
    n = ori_num * mul
    new_data = np.zeros((n, length))
    for i in range(ori_num):
        for j in range(mul):
            new_data[i * mul + j, 0] = data[i, 0]
            new_data[i * mul + j, 1:] = data[i, j + 1:j + length]
    return new_data


def _normalize(data):
    mean = data.mean(axis=1).reshape((data.shape[0], 1))
    std = data.std(axis=1).reshape((data.shape[0], 1))
    return (data - mean) / std


def get_data(train_path, test_path, valid_id=0, isNorm=True, length=1024):
    train = _load_data(train_path)
    test = _load_data(test_path)
    train_num = train.shape[0]
    if valid_id > 0:
        idn = np.arange(train_num)
        valid = train[idn[train_num *
                          (valid_id - 1) // 5:train_num * valid_id // 5]]
        idn = np.delete(idn, range(train_num * (valid_id - 1) //
                                   5, train_num * valid_id // 5))
        train = train[idn]
    else:
        valid = train.copy()
    train = _slice_data(train, length)
    valid = _slice_data(valid, length)
    test = _slice_data(test, length)
    if isNorm:
        train[:, 1:] = _normalize(train[:, 1:])
        valid[:, 1:] = _normalize(valid[:, 1:])
        test[:, 1:] = _normalize(test[:, 1:])
    return ((np.int_(train[:, 0] - 1), train[:, 1:]),
            (np.int_(valid[:, 0] - 1), valid[:, 1:]), (np.int_(test[:, 0] - 1), test[:, 1:]))


if __name__ == '__main__':
    data = get_data('./test_data', './test_data', valid_id=2, isNorm=True)
    train_label, train_data = data[0]
    valid_label, valid_data = data[1]
    test_label, test_data = data[2]
    print(train_label)
    print(train_data)
    print('--------------------------------')
    print(valid_label)
    print(valid_data)
    print('--------------------------------')
    print(test_label)
    print(test_data)
