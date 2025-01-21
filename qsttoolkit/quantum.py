import numpy as np
from scipy.linalg import sqrtm
from qutip import Qobj


##### Quantum physics #####

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Returns the fidelity between two density matrices.
    
    Parameters
    ----------
    rho : np.ndarray
        The first density matrix.
    sigma : np.ndarray
        The second density matrix.

    Returns
    -------
    float
        The fidelity between the two density matrices.
    """
    sqrt_sigma = sqrtm(sigma)
    return np.real(np.trace(sqrtm(sqrt_sigma @ rho @ sqrt_sigma))**2)


##### General density matrices - initial guesses for MLE #####

def maximally_mixed_state_dm(dim: int) -> Qobj:
    """
    Returns the maximally mixed state density matrix.
    
    Parameters
    ----------
    dim : int
        The Hilbert space dimensionality.
    
    Returns
    -------
    Qobj
        The maximally mixed state density matrix.
    """
    return Qobj(np.eye(dim) / dim)

def random_positive_semidefinite_dm(dim: int) -> Qobj:
    """
    Returns a random positive semi-definite density matrix.
    
    Parameters
    ----------
    dim : int
        The Hilbert space dimensionality.

    Returns
    -------
    Qobj
        A random positive semi-definite density matrix.
    """
    random_matrix = np.random.rand(dim, dim)
    return Qobj(random_matrix @ random_matrix.T)