from keras.models import Sequential

from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D,Flatten


def model1():
    model = Sequential()
    model.add(Convolution2D(nb_filter=10,nb_row=5,nb_col=7,input_shape=(600,600,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(nb_filter=10,nb_row=5,nb_col=7,activation='relu'))
    model.add(Convolution2D(nb_filter=20,nb_row=5,nb_col=7,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(output_dim=50,activation='relu'))
    model.add(Dense(output_dim=8,activation='softmax'))

    return model

