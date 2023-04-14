import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    distance = np.linalg.norm(a - b, axis=-1)
    return distance if isinstance(distance, np.ndarray) else np.array([distance])


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist
