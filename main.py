import numpy as np
from capa import Capa
from mlp import MLP

if __name__ == "__main__":
    x = np.array([
        [0.1, 0.2, 0.3],
        [0.5, -0.2, 0.1],
        [-0.3, 0.8, -0.5],
        [0.0, 0.0, 0.0]
    ])
    red = MLP([
        Capa(num_entradas=3, num_neuronas=4, activacion=1 / (1 + np.exp(-x))),
        Capa(num_entradas=4, num_neuronas=3, activacion=1 / (1 + np.exp(-x))),
        Capa(num_entradas=3, num_neuronas=1, activacion=np.maximum(0, x))
    ])
    y_pred = red.predecir(X)
    print("Predicciones:\n", y_pred)
