import numpy as np
from typing import List
from capa import Capa

class MLP:
    def __init__(self, capas: List[Capa]):
        self.capas = capas

    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        salida = entradas
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida
