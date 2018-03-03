import numpy as np
import model


data = model.get_data()
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]
model = model.Model(n, k, wh)
#model.try_load()

x_train, y_train, x_test, y_test = model.load_data()

for lr in [0.1, 0.01, 0.001]
    model.train(x_train, y_train, x_test, y_test, batch_size=100, learning_late=lr)