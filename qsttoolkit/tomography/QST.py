import numpy as np
from qutip import coherent, fock


##### Define parameterisation functions #####

def parameterise_density_matrix(rho: np.ndarray) -> np.ndarray:
    """
    Parameterises the density matrix using the Cholesky decomposition.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix to be parameterised.

    Returns
    -------
    np.ndarray
        Flattened vector of real parameters.
    """
    initial_cholesky = np.linalg.cholesky(rho)
    initial_params = []
    for i in range(rho.shape[0]):
        for j in range(i + 1):
            initial_params.append(initial_cholesky[i, j].real)
            if i != j:
                initial_params.append(initial_cholesky[i, j].imag)
    return np.array(initial_params)

def reconstruct_density_matrix(params: np.array, dim: int) -> np.ndarray:
    """
    Reconstructs the density matrix from the flattened vector of real parameters.
    
    Parameters
    ----------
    params : np.array
        Flattened vector of real parameters.
    dim : int
        The Hilbert space dimensionality.

    Returns
    -------
    np.ndarray
        Reconstructed density matrix.
    """
    T = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            # Real part
            T[i, j] = params[idx]
            idx += 1
            # Imaginary part (only for off-diagonal elements)
            if i != j:
                T[i, j] += 1j * params[idx]
                idx += 1
    # Construct the density matrix - invert the Cholesky decomposition
    rho = T.conj().T @ T
    # Normalize to trace 1
    rho /= np.trace(rho)
    return rho


##### Define measurement operators #####
### Specific measurement operators - to be removed ###

def Husimi_Q_measurement_operators(dim: int, xgrid: np.array, pgrid: np.array) -> np.array:
    """Returns the measurement operators for the Husimi-Q function (projectors of all possible coherent operators created from the phase space provided by xgrid and pgrid)."""
    E = []
    for x in xgrid:
        for p in pgrid:
            E.append(np.outer(coherent(dim, x + 1j*p).full(), coherent(dim, x + 1j*p).full().conj().T))
    return np.array(E)

def photon_number_measurement_operators(dim: int) -> np.array:
    """Returns the measurement operators for the photon number measurement."""
    E = []
    for n in range(dim):
        E.append(np.outer(fock(dim, n).full(), fock(dim, n).full().conj().T))
    return np.array(E)


### Generalised measurement operators ###

def measurement_operators(dim: int, measurement_type: str, **kwargs) -> np.array:
    """
    Returns the measurement operators for the specified measurement type.

    Parameters
    ----------
    dim : int
        The Hilbert space dimensionality.
    measurement_type : str
        The type of measurement to be performed.
    **kwargs : dict
        Additional keyword arguments required for specific measurement types.

    Returns
    -------
    np.array
        Measurement operators.
    """
    E = []
    if measurement_type == 'Husimi-Q':
        if 'xgrid' not in kwargs or 'pgrid' not in kwargs:
            raise ValueError("For Husimi-Q measurement, xgrid and pgrid must be provided.")
        for x in kwargs['xgrid']:
            for p in kwargs['pgrid']:
                E.append(np.outer(coherent(dim, x + 1j*p).full(), coherent(dim, x + 1j*p).full().conj().T))
    elif measurement_type == 'photon_number':
        if 'dim_limit' in kwargs:
            dim = kwargs['dim_limit']
        for n in range(dim):
            E.append(np.outer(fock(dim, n).full(), fock(dim, n).full().conj().T))
    else:
        raise ValueError(f"Measurement type {measurement_type} not recognised.")
    return np.array(E)


##### Define constraints #####

def trace_constraint(params: np.array, dim: int) -> float:
    """
    Constraint function to ensure the trace of the density matrix is 1.
    
    Parameters
    ----------
    params : np.array
        Flattened vector of real parameters.
    dim : int
        The Hilbert space dimensionality.

    Returns
    -------
    float
        The difference between the trace of the reconstructed density matrix and 1.
    """
    rho = reconstruct_density_matrix(params, dim)
    return np.trace(rho).real - 1  # Should be zero

def positivity_constraint(params: np.array, dim: int) -> float:
    """
    Constraint to ensure the density matrix is positive semi-definite - ensures the smallest eigenvalue of rho is non-negative.
    
    Parameters
    ----------
    params : np.array
        Flattened vector of real parameters.
    dim : int
        The Hilbert space dimensionality.

    Returns
    -------
    float
        The smallest eigenvalue of the reconstructed density matrix.
    """
    rho = reconstruct_density_matrix(params, dim)
    eigenvalues = np.linalg.eigvalsh(rho)  # Eigenvalues of rho
    return np.min(eigenvalues)  # Should be >= 0