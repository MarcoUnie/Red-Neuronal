import tensorflow as tf

def compile_model((input_shape=(784,)):
    model = tf.keras.Sequential(tf.keras.layers.Dense(512, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"))
        keras.datasets
    return model
