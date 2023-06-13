import keras.api._v2.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from scikeras.wrappers import KerasClassifier
from dataLoad import plot_acc
import tensorflow as tf
from keras.utils.np_utils import to_categorical 



def CNNClassifier(train_inputs, train_labels, test_inputs, test_labels):
    model = Sequential()
    model.add(Reshape((32, 32, 3)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))

    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Conv2D(20, (3, 3), padding='same'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))

    keras.layers.Dropout(0.3, noise_shape=None, seed=None)



    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)


    model.compile(loss='categorical_crossentropy',
                    optimizer = 'adam', 
                    metrics = ['accuracy'])
    #model.predict(cd_test_inputs[:1])

    history = model.fit(train_inputs, train_labels, \
                        validation_data=(test_inputs, test_labels), \
                        epochs=10)
    plot_acc(history)
