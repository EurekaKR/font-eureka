from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import json

p = Path('.')
model_dir = p / 'models'

class Model(object):
    def __init__(self):
        self.model = None
        # self.dataset_manifest = None

    def build(self, shape, batch_size, learning_late=0.1, epochs=1):
        model = Sequential()

        #Conv2D need 4D shape
        model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=(shape[1], shape[2], 1)))
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
       
        self.model = model

    def train(self, x_train, y_train):
        if self.model is None:
            print("please build or load model")
            raise ValueError
        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,nverbose=1)
        
    def load_model(self, model_path):
        # TODO: how to check error in load_model
        self.model = load_model(model_path)

    def test(self, x_test, y_test):
        model.evaluate(x_test, y_test, verbose=1)
        print(result)
        return result

    def save_model(self, model_name):
        if not model_dir.exists():
            model_dir.mkdir()
        
        model_path = model_dir / model_name
        if model_path.exists():
            print("alread exist filename")
            raise

        self.model.save(f"{model_path}.h5")
        with open(f"{model_path}.json", 'w') as mmp:
            json.dump({
                '_type': 'model_manifest',
                '_version': 1,
                'name': model_name,
                #'using_dataset_name': self.dataset_manifest,
            }, mmp)
        print(f"save to {model_path}.h5")

if __name__ == '__main__':
    x_input, y_input, n = get_data('fonts.hdf5')
    model = Model(n)

    x_train, y_train, x_test, y_test = load_data(x_input, y_input)

    for lr in [0.1, 0.01, 0.001]:
        model.train(x_train, y_train, x_test, y_test, batch_size=100, learning_late=lr)
        model.model_test(x_test, y_test)

    '''
    x_input, y_input = get_data('fonts.hdf5')
    CNN_model = Model()
    # if want to build new model then    
    CNN_model.build(x_input.shape, batch_size, learning_late, epochs)
    CNN_model.train(x_input, y_input)
    # if load exist model then
    CNN_model.load_model(model_path)

    CNN_model.test(x_test, y_test)

    # if want to save model
    CNN_model.save_model(model_name) # save to models/model_name.h5
    '''
