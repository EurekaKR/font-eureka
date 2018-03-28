import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import h5py
import numpy as np


RESHAPER = lambda a: a.reshape((a.shape[0], a.shape[1], 
                                a.shape[2], 1))

def get_data():
    f = h5py.File('fonts.hdf5', 'r')
    data = np.array(f['fonts'])
    n, k, w, h = data.shape
    x_input = data.reshape(-1, w, h)
    #x_input = x_input.reshape(n * k, w, h, 1)
    y_input = np.zeros((n * k, n))
    print(x_input.shape)
    for i in range(n):
        for j in range(k):
            y_input[(i * k) + j][i] = 1
    return x_input, y_input, n, k, w * h



def load_data(x_input, y_input):
    x_train, x_test, y_train, y_test = train_test_split(
        x_input, y_input, test_size=0.33, random_state=42)
    print('x_train', x_train.shape)
    print('x_test', x_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)
    return x_train, y_train, x_test, y_test


class Model(object):
    def __init__(self, n=None, k=11173, wh=75 * 75, d=40, D=1024):
        self.n, self.k, self.d, self.D = n, k, d, D
        self.model = None
        self.num_classes = n

    def train(self, x_train, y_train, x_test, y_test, batch_size, learning_late=0.1, epochs=1):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
            activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dense(self.D))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(3, kernel_size=(3, 3),
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Conv2D(3, kernel_size=(3, 3),
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        x_train = RESHAPER(x_train)
       
        print("x_train shape", x_train.shape, "y_train shape", y_train.shape)
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        self.model = model
        return model

    def model_test(self, x_test, y_test):
        x_test = RESHAPER(x_test)
        result = self.model.evaluate(x_test, y_test, verbose=1)
        print(result)
        return result

    def save_model(self):
        print('saving model...')
        self.model.save('font_eureka_model.h5')

