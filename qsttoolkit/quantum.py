import numpy as np
from scipy.linalg import sqrtm
from qutip import Qobj
import tensorflow as tf


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

def expectation(rho: tf.Tensor, measurement_operators: list[tf.Tensor]) -> tf.Tensor:
    """
    Computes the expectation values of the given density matrix with respect to the given projective measurement operators using purely TensorFlow operations.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix to compute expectation values for.
    measurement_operators : list of tf.Tensor
        Projective measurement operators to compute the expectation values for.

    Returns
    -------
    tf.Tensor
        Expectation values of the density matrix with respect to the measurement operators.
    """
    if type(rho) == Qobj:
        rho = rho.full()

    measurements = [tf.linalg.trace(tf.matmul(E, rho)) for E in measurement_operators]
    norm_real_measurements = tf.linalg.normalize(tf.math.real(measurements))[0]
    return tf.reshape(norm_real_measurements, (1, len(norm_real_measurements)))


##### General density matrices - initial guesses for MLE #####

def maximally_mixed_state_dm(dim: int) -> Qobj:
    """
    Computes the maximally mixed state density matrix in the given Hilbert space dimensionality.
    
    Parameters
    ----------
    dim : int
        Hilbert space dimensionality.
    
    Returns
    -------
    Qobj
        Maximally mixed state density matrix.
    """
    if not isinstance(dim, int): raise ValueError("dim must be an integer.")

    return Qobj(np.eye(dim) / dim)

def random_positive_semidefinite_dm(dim: int) -> Qobj:
    """
    Computes a random positive semi-definite density matrix in the given Hilbert space dimensionality.
    
    Parameters
    ----------
    dim : int
        Hilbert space dimensionality.

    Returns
    -------
    Qobj
        Random positive semi-definite density matrix.
    """
    if not isinstance(dim, int): raise ValueError("dim must be an integer.")

    random_matrix = np.random.rand(dim, dim)
    Hermitian = random_matrix @ random_matrix.T
    return Qobj(Hermitian / np.trace(Hermitian))