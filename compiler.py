import tensorflow as tf

def compile_model(architecture_string: str, input_shape=None):
    """
    Compila un modelo Keras a partir de una cadena de arquitectura.
    Ej: "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
    """
    model = tf.keras.Sequential()
    capas = [capa.strip() for capa in architecture_string.split("->")]

    for i, capa in enumerate(capas):
        if not capa.startswith("Dense"):
            raise ValueError(f"Tipo de capa no soportado: {capa}")

        # Obtener contenido entre paréntesis
        contenido = capa[capa.find("(")+1 : capa.find(")")]
        partes = [x.strip() for x in contenido.split(",")]

        if len(partes) != 2:
            raise ValueError(f"Formato inválido en capa: {capa}. Usa Dense(units, activation)")

        unidades = int(partes[0])
        activacion = partes[1]

        # Si es la primera capa y hay input_shape, se añade explícitamente
        if i == 0 and input_shape is not None:
            model.add(tf.keras.layers.Dense(unidades, activation=activacion, input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Dense(unidades, activation=activacion))

    return model
