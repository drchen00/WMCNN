#!/usr/bin/python
import sys
import numpy as np

def load_data(path):
    data = np.loadtxt(path,delimiter=',')
    matrix = data[:,1:]
    label = data[:,0]
    return matrix,label

if __name__ == '__main__':
    matrix,label=load_data('../UCR_TS_Archive_2015/50words/50words_TRAIN')
    print(matrix.shape) 

