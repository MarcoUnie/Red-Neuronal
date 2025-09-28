import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from compiler import compile_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 784)).astype("float32") / 255
x_test = x_test.reshape((-1, 784)).astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

modelo = compile_model(input_shape=(784,))

modelo.compile(optimizer="adam",
               loss="categorical_crossentropy",
               metrics=["accuracy"])
               
modelo.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)

loss, acc = modelo.evaluate(x_test, y_test, verbose=0)
modelo.save("model.h5")
