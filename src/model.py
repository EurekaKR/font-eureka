from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import os


RESHAPER = lambda a: a.reshape((a.shape[0], a.shape[1], a.shape[2], 1))  # TODO: PULL THIS OUT TO create_dataset.py

 # DEPENDENCY: create_dataset.py
 def get_data(name='fonts.hdf5'):
    f = h5py.File(name, 'r')
    data = np.array(f['fonts'])
    n, k, w, h = data.shape 
    x_input = data.reshape(-1, w, h)
    y_input = np.zeros((n * k, n))
    for i in range(n):
        for j in range(k):
            y_input[(i * k) + j][i] = 1
    return x_input, y_input, n


# TODO: create two dataset
# dataset/train, dataset/test
def load_data(x_input, y_input):
    x_train, x_test, y_train, y_test = train_test_split(
        x_input, y_input, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test

class Model(object):

    def __init__(self, n=None):
        self.num_classes = n
        # self.dataset_manifest = None

    def train(self, x_train, y_train, x_test, y_test, batch_size, learning_late=0.1, epochs=1):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
            activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dense(1024))

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

    def save_model(self, model_name):
        print('saving model...')
        if not os.exists('models'):  # TODO: pathlib.
            os.mkdir('models')
        self.model.save('models/%s.h5' % (model_name,))
        with open('models/%s.json' % (model_name,), 'w') as mmp:
            import json
            json.dump({
                '_type': 'model_manifest',
                '_version': 1,
                'name': model_name,
                #'using_dataset_name': self.dataset_manifest,
            }, mmp)

if __name__ == '__main__':
    x_input, y_input, n = get_data('fonts.hdf5')
    model = Model(n)

    x_train, y_train, x_test, y_test = load_data(x_input, y_input)

    for lr in [0.1, 0.01, 0.001]:
        model.train(x_train, y_train, x_test, y_test, batch_size=100, learning_late=lr)
        model.model_test(x_test, y_test)
