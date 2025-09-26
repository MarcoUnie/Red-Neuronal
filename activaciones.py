import numpy as np
def sigmoide(x: np.ndarray) -> np.ndarray:
    """Función de activación Sigmoide."""
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """Función de activación ReLU."""
    return np.maximum(0, x)