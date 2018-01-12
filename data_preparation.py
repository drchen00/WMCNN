#!/usr/bin/python
import numpy as np

def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    rng = np.random.RandomState(12345)
    ind = np.arange(data.shape[0])
    rng.shuffle(ind)
    data = data[ind]
    matrix = data[:, 1:]
    label = data[:, 0]
    return matrix, label

if __name__ == '__main__':
    matrix, label = load_data('../UCR_TS_Archive_2015/CBF/CBF_TRAIN')
    n = matrix.shape[0]
    print(matrix)
