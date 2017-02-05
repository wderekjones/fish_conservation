from keras.models import Sequential

from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D,Flatten
from keras.layers.advanced_activations import PReLU


def model1():
    model = Sequential()
    model.add(Convolution2D(nb_filter=30,nb_row=5,nb_col=7,input_shape=(600,600,1),activation=PReLU()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(nb_filter=10,nb_row=5,nb_col=7,activation=PReLU()))
    model.add(Convolution2D(nb_filter=20,nb_row=5,nb_col=7,activation=PReLU()))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(output_dim=10,activation=PReLU()))
    model.add(Dense(output_dim=8,activation='softmax'))

    return model

