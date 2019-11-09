import six.moves.cPickle as pickle
import glob
import os
import numpy as np
import re
from PIL import Image

def load_data(dataset, train_data):
    ''' 
    Loads the dataset
    '''
    train_set = []
    test_set = []
    cur_path = os.path.join(os.getcwd(), dataset)
    if train_data == 'cis':
        dataset_idx = input('[cis_train idx (1 or 2)]: ')
        dataset_tr = 'cis_train' + dataset_idx + '.txt'
        dataset_test = 'cis_test.txt'
        with open(os.path.join(cur_path, dataset_tr), 'r') as f:
            for line in f:
                train_d = line.strip().split()
                train_set.append([float(i) for i in train_d])
        train_set = np.array(train_set)
        train_y = train_set[:, len(train_set[0])-1] # Labels only
        train_x = np.delete(train_set, len(train_set[0])-1, 1)
        train_set = (train_x, train_y)

        with open(os.path.join(cur_path, dataset_test), 'r') as f:
            for line in f:
                test_d = line.strip().split()
                test_set.append([float(i) for i in test_d])
        test_set = np.array(test_set)
        test_y = test_set[:, len(test_set[0])-1] # Labels only
        test_x = np.delete(test_set, len(test_set[0])-1, 1)
        test_set = (test_x, test_y)

    elif train_data == 'fa':
        dataset_idx = input('[fa_train idx (1 or 2)]: ')
        dataset_tr = 'fa_train' + dataset_idx + '.txt'
        dataset_test = 'fa_test.txt'
        with open(os.path.join(cur_path, dataset_tr), 'r') as f:
            for line in f:
                train_d = line.strip().split()
                train_set.append([float(i) for i in train_d])
        train_set = np.array(train_set)
        train_set = train_set[train_set[:,0].argsort()]
        train_y = train_set[:, len(train_set[0])-1] # Labels only
        train_x = np.delete(train_set, len(train_set[0])-1, 1)
        train_set = (train_x, train_y)

        with open(os.path.join(cur_path, dataset_test), 'r') as f:
            for line in f:
                test_d = line.strip().split()
                test_set.append([float(i) for i in test_d])
        test_set = np.array(test_set)
        test_set = test_set[test_set[:,0].argsort()]
        test_y = test_set[:, len(test_set[0])-1] # Labels only
        test_x = np.delete(test_set, len(test_set[0])-1, 1)
        test_set = (test_x, test_y)

    return train_set, test_set