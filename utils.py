import os

import skimage as sk

import numpy as np

from sklearn import preprocessing

from scipy.ndimage import imread


path = 'train/train/'


dirnames = []

filenames = []


for root, dirs, files in os.walk(path):
    if len(files) > 1:
        filenames.append(files)
    if len(dirs) > 0:
        dirnames = dirs



fish_labels = np.ndarray([])
fish_data = np.ndarray([])


for i in range(0,len(dirnames)):

    # iterate over and load each file/label
    path_len = len(filenames[i])

    labels = i * (np.ones([path_len,1]))
    for f in filenames[i]:
        relpath = path+dirnames[i]+'/'+f
        # now read the images and store the labels

        imread(relpath)