import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from compiler import compile_model
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 784)).astype("float32") / 255
x_test = x_test.reshape((-1, 784)).astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

modelo = compile_model(input_shape=(784,))

modelo.compile(optimizer="adam",
               loss="categorical_crossentropy",
               metrics=["accuracy"])
               
history = modelo.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=128,
    verbose=1
)


plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("resultados/perdidas.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("resultados/precision.png")
plt.close()

loss, acc = modelo.evaluate(x_test, y_test, verbose=0)
with open("resultados/precision_final.txt", "w") as f:
    f.write(f"Precisión final en test: {acc*100:.2f}%\n")
modelo.save("model.h5")
