import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import h5py

def load_data(x_input, y_input):
    x_train, y_train, x_test, y_test = train_test_split(
        x_input, y_input, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test


def get_data():
    f = h5py.File('fonts.hdf5', 'r')
    return f['fonts']


class Model(object):
    def __init__(self, n=None, k=11173, wh=75 * 75, d=40, D=1024):
        self.n, self.k, self.d = n, k, d
        self.model = None
        self.num_classes = n

    def train(self, x_train, y_train, x_test, y_test, batch_size, learning_late=0.1, epochs=10):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', input_shape=x_train.shape[1:]))
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
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_classes, activation='relu'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        return model

    @staticmethod
    def model_test(model, x_test, y_test):
        result = model.evaluate(x_test, y_test, verbose=0)
        return result

    @staticmethod
    def save(model):
        print('saving model...')
        model.save('font_eureka_model.h5')

