import numpy as np
from scipy.linalg import sqrtm
from qutip import Qobj
import tensorflow as tf

from qsttoolkit.utils import _deprecation_warning


##### Quantum physics #####

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Computes the fidelity between two density matrices.
    
    Parameters
    ----------
    rho : np.ndarray
        First density matrix.
    sigma : np.ndarray
        Second density matrix.

    Returns
    -------
    float
        Fidelity between the two density matrices.
    """
    if type(rho) == Qobj:
        rho = rho.full()
    elif type(rho) == np.ndarray:
        pass
    else:
        rho = rho.numpy()
    if type(sigma) == Qobj:
        sigma = sigma.full()
    elif type(sigma) == np.ndarray:
        pass
    else:
        sigma = sigma.numpy()
    
    sqrt_sigma = sqrtm(sigma)
    return np.real(np.trace(sqrtm(sqrt_sigma @ rho @ sqrt_sigma))**2)

def expectation(rho: tf.Tensor, measurement_operators: list[Qobj], numpy: bool=False) -> tf.Tensor:
    """
    Computes the expectation values of the given density matrix with respect to the given projective measurement operators using purely TensorFlow operations.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix to compute expectation values for.
    measurement_operators : list of Qobj
        Projective measurement operators to compute the expectation values for.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to False.

    Returns
    -------
    tf.Tensor
        Expectation values of the density matrix with respect to the measurement operators.
    """
    if type(rho) == Qobj:
        rho = rho.full()
    if type(measurement_operators[0]) == Qobj:
        measurement_operators = [E.full() for E in measurement_operators]

    measurements = [tf.linalg.trace(tf.matmul(E, rho)) for E in measurement_operators]
    norm_real_measurements = tf.linalg.normalize(tf.math.real(measurements))[0]
    reshaped_measurements = tf.reshape(norm_real_measurements, (1, len(norm_real_measurements)))
    if numpy:
        return reshaped_measurements.numpy().flatten()
    else:
        return reshaped_measurements

def hadamard() -> Qobj:
    """
    Returns the single-qubit Hadamard gate.

    Returns
    -------
    Qobj
        Hadamard gate as a Qobj.
    """
    return Qobj([[1, 1], [1, -1]]) / np.sqrt(2)

def phase_space_grid(x_min: float, x_max: float, p_min: float, p_max: float, num_x_points: int, num_p_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a grid of points in x-p phase space.

    Parameters
    ----------
    x_min : float
        Minimum x-coordinate of the grid.
    x_max : float
        Maximum x-coordinate of the grid.
    p_min : float
        Minimum p-coordinate of the grid.
    p_max : float
        Maximum p-coordinate of the grid.
    num_x_points : int
        Number of points along the x-axis.
    num_p_points : int
        Number of points along the p-axis.

    Returns
    -------
    np.ndarray
        2D array containing the complex coordinates of the grid points in x-p phase space.
    """
    x = np.linspace(x_min, x_max, num_x_points)
    p = np.linspace(p_min, p_max, num_p_points)
    
    X, P = np.meshgrid(x, p)
    return X + 1j * P

##### General density matrices - initial ansatzes for MLE #####

def maximally_mixed_state_dm(N: int, dim=None) -> Qobj:
    """
    Computes the maximally mixed state density matrix in the given Hilbert space dimensionality.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    
    Returns
    -------
    Qobj
        Maximally mixed state density matrix.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim
    if not isinstance(N, int): raise ValueError("N must be an integer.")

    return Qobj(np.eye(N) / N)

def random_positive_semidefinite_dm(N: int, dim=None) -> Qobj:
    """
    Computes a random positive semi-definite density matrix in the given Hilbert space dimensionality.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.

    Returns
    -------
    Qobj
        Random positive semi-definite density matrix.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim
    if not isinstance(N, int): raise ValueError("N must be an integer.")
    
    random_matrix = np.random.rand(N, N)
    Hermitian = random_matrix @ random_matrix.T
    return Qobj(Hermitian / np.trace(Hermitian))