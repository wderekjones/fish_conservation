import os

import skimage as sk

import numpy as np

from sklearn import preprocessing

from scipy.ndimage import imread

from scipy.misc import imresize
import matplotlib.pyplot as plt

import random


def load_from_folder(path,folder,batch_size):

    '''

        This function reads a bounded number of images from a particular folder. Returns a numpy array of pixel data and a numpy of labels.

    :param path:
    :param folder:
    :param batch_size:
    :return:
    '''

    path = path + '/' + folder
    label_dict = {'ALB': 0, 'BET': 1, 'DOL': 2, 'LAG': 3, 'NoF': 4, 'OTHER': 5, 'SHARK': 6, 'YFT': 7}

    l = label_dict[folder]

    print 'value of labels before any computation: '+str(l)

    filenames = []



    num_examples = 0

    for root, dirs, files in os.walk(path):
        num_examples = num_examples + len(files)
        if len(files) > 1:
            #filenames.append(files)
            filenames = files

    fish_labels = np.ndarray([batch_size, 1])
    fish_data = np.ndarray([batch_size, 600, 600, 3])
    j = 0

    for file in filenames:


        # now read the images and store the labels

        if j < batch_size:
            relpath = path + '/' + file
            image = imread(relpath)

            image = imresize(image,[600,600])
            fish_data[j,:,:,:] = image
            fish_labels[j] = l
            print l
            j = j + 1
        else:
            return fish_data, fish_labels

    return fish_data, fish_labels



def load_batch(max_ex_per_cat):
    '''
        This function iterates over each folder and pulls a number of samples (max_ex_per_cat, or maximum examples per category),
        from each category in order to form a training batch.

    '''

    batch_data = np.zeros([0,600,600,3])
    batch_labels = np.zeros([0,1])

    dirs = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

    for d in dirs:
        x,y = load_from_folder('train/train',d,max_ex_per_cat)
        batch_data = np.append(batch_data,x, axis=0)
        batch_labels = np.append(batch_labels,y,axis=0)
    return batch_data, batch_labels

X_train,y_train = load_batch(2)

