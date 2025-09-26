import numpy as np
from typing import Callable
from neurona import Neurona

class Capa:
    def __init__(self, num_entradas: int, num_neuronas: int, activacion: Callable):
        self.neuronas = [
            Neurona(
                pesos=np.random.randn(num_entradas),
                sesgo=np.random.randn(),
                activacion=activacion
            )
            for _ in range(num_neuronas)
        ]

    def forward(self, entradas: np.ndarray) -> np.ndarray:
        salidas = []
        for muestra in entradas:
            salidas.append([neurona.forward(muestra) for neurona in self.neuronas])
        return np.array(salidas)
