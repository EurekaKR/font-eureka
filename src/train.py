import numpy as np
import model as md


x_input, y_input, n, k, wh = md.get_data()
model = md.Model(n, k, wh)

print(n, k, wh)
#model.try_load()

x_train, y_train, x_test, y_test = md.load_data(x_input,
        y_input)

for lr in [0.1, 0.01, 0.001]:
    model.train(x_train, y_train, x_test, y_test, batch_size=100, learning_late=lr)
