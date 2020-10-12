# -*- coding:utf-8 -*-

import os
import random
import numpy as np
from skimage import io, transform

label_map = {'calling_images': 0, 'smoking_images': 1, 'normal_images': 2}

def load_min_batch(data_list, label_list, height=240, width=320):
    assert len(data_list) == len(label_list)
    images = []
    labels = []
    for index in range(len(data_list)):
        image = io.imread(data_list[index])
        image = transform.resize(image, (height, width))
        images.append(image)
        labels.append(label_map[label_list[index]])
    images = np.asarray(images, np.float32)
    labels = np.asarray(labels, np.int32)
    return images, labels

def load_file(dir_path):
    datas = []
    labels = []
    for dir_name in os.listdir(dir_path):
        dir = os.path.join(dir_path, dir_name)
        if os.path.isdir(dir):
            for image in os.listdir(dir):
                datas.append(os.path.join(dir, image))
                labels.append(dir_name)
        elif os.path.isfile(dir_name):
            print("This is a normal file")
            continue
        else:
            print("This is a special file")
            continue
    return datas, labels

def shuffle_data(datas, labels):
    num_data = len(datas)
    arr = np.arange(num_data)
    np.random.shuffle(arr)
    datas = np.array(datas)[arr]
    labels = np.array(labels)[arr]
    return datas, labels

def prepare_data(dataset_path):
    print("======Loading data======")
    train_dataset_path = os.path.join(dataset_path, 'train')
    test_dataset_path = os.path.join(dataset_path, 'test')
    train_datas, train_labels = load_file(train_dataset_path)
    test_datas, test_labels = load_file(test_dataset_path)

    print("======Shuffling data======")
    #shuffle_data(train_datas, train_labels)
    indices = np.random.permutation(len(train_datas))
    train_datas = np.array(train_datas)[indices]
    train_labels = np.array(train_labels)[indices]

    return train_datas, train_labels, test_datas, test_labels

def one_hot(labels, num_classes):
    n_sample = len(labels)
    n_class = num_classes
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels
