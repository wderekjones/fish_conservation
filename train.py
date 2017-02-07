import numpy as np
from models import*

from utils import *

import time
import h5py


from sklearn.model_selection import train_test_split

from skimage import feature

from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import matplotlib.pyplot as plt




X, y = load_batch(10)

y = to_categorical(y,8)

X = compute_canny(X,3)

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)


date = time.strftime("%m:%d:%Y")

time = time.strftime('%H:%M:%S', time.gmtime())

model_name = "checkpoints/model"+date+time

model_file = open(model_name,mode='w')


checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss')

model = model1()

model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X,y,batch_size=10,nb_epoch=100,shuffle=True,validation_split=0.2, callbacks=[checkpointer])

model_file.close()
