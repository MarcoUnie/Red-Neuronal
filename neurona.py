import numpy as np
from typing import Callable

class Neurona:
    def __init__(self, pesos: np.ndarray, sesgo: float, activacion: Callable):
        self.pesos = pesos
        self.sesgo = sesgo
        self.activacion = activacion

    def forward(self, entradas: np.ndarray) -> float:
        z = np.dot(entradas, self.pesos) + self.sesgo
        return self.activacion(z)
