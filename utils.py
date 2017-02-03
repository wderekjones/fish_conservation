import os

import skimage as sk

import numpy as np

from sklearn import preprocessing

from scipy.ndimage import imread

import matplotlib.pyplot as plt

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

        image = imread(relpath)
        print image.shape

        '''hmmm.... either pad the images with 0's in order to conform to a single size...(but the images get a bit large afterwards) or

            downsample the images in order to create smaller images of the same size...

            ALSO: consider performing edge detection to extract relevant information from the image

            '''
