from keras.models import load_model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras


def load_data():
    x_train, y_train, x_test, y_test = None, None, None, None
    return x_train, y_train, x_test, y_test


class Model(object):
    def __init__(self, n=None, k=11173, wh=64 * 64, d=40, D=1024, batch_size=512):
        self.n, self.k, self.d = n, k, d
        self.model = None
        self.num_classes = n

        self.x_train, self.y_train, \
            self.x_test, self.y_test = load_data()

    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs=10):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', input_shape=x_train.shape[1:])
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