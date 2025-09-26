import numpy as np
from typing import List
from capa import Capa

class MLP:
    def __init__(self, capas: List[Capa]):
        """Inicializa el MLP con una lista de capas."""
        self.capas = capas

    def predecir(self, entradas: np.ndarray) -> np.ndarray:
        """Realiza la propagación hacia adelante en todas las capas."""
        salida = entradas
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida
