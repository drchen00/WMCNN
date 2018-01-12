#!/usr/bin/python
import numpy as np


def _load_data(path):
    data = np.loadtxt(path, delimiter=',')
    rng = np.random.RandomState(12345)
    ind = np.arange(data.shape[0])
    rng.shuffle(ind)
    data = data[ind]
    matrix = data[:, 1:]
    label = data[:, 0]
    return matrix, label


def slice_data(path, length=1024):
    matrix, label = _load_data(path)
    ori_num = matrix.shape[0]
    ori_len = matrix.shape[1]
    threshold = min(ori_len, length) / 2
    length = 1
    while length <= threshold:
        length <<= 1
    mul = ori_len - length + 1
    n = ori_num * mul
    new_matrix = np.zeros((n, length))
    new_label = np.zeros(n)
    for i in range(ori_num):
        for j in range(mul):
            new_matrix[i * mul + j, :] = matrix[i, j:j + length]
            new_label[i * mul + j] = np.int_(label[i])
    return new_matrix, new_label


if __name__ == '__main__':
    matrix, label = slice_data('./test_data', 3)
    print(matrix)
    print(label)
