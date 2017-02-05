import numpy as np
from models import model1

from utils import *


from sklearn.model_selection import train_test_split

from skimage import feature

import matplotlib.pyplot as plt




X, y = load_batch(10)

y = to_categorical(y,8)

X = compute_canny(X,3)

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)



model = model1()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X,y,batch_size=10,nb_epoch=100,shuffle=True)

