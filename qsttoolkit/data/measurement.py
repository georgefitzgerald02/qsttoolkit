import numpy as np
from qutip import coherent, fock, displace

from qsttoolkit.utils import _deprecation_warning


##### Define measurement operators #####
### Specific measurement operators ###

def heterodyne_measurement_operators(N: int, xgrid: np.ndarray=None, pgrid: np.ndarray=None, alpha_grid: np.ndarray=None, numpy: bool=True) -> np.ndarray:
    """
    Computes the measurement operators for heterodyne detection of an optical quantum state across a grid of phase space displacements.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    xgrid : np.ndarray
        Phase space X quadrature grid.
    pgrid : np.ndarray
        Phase space P quadrature grid.
    alpha_grid : np.ndarray
        Complex amplitude grid. If provided, xgrid and pgrid are ignored.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to True.

    Returns
    -------
    np.ndarray
        Measurement operators.
    """
    if not isinstance(N, int): raise ValueError("N must be an integer.")
    if xgrid is None and pgrid is None and alpha_grid is None: raise ValueError("Either xgrid and pgrid, or alpha_grid, must be provided.")
    
    if alpha_grid is None:
        if (xgrid is not None and pgrid is None) or (xgrid is None and pgrid is not None): raise ValueError("Both xgrid and pgrid must be provided together.")
        if not isinstance(xgrid, np.ndarray) or not isinstance(pgrid, np.ndarray): raise ValueError("xgrid and pgrid must be numpy arrays.")
        X, P = np.meshgrid(xgrid, pgrid)
        alpha_grid = X + 1j * P
    else:
        if not isinstance(alpha_grid, np.ndarray): raise ValueError("alpha_grid must be a numpy array.")
        if alpha_grid.ndim != 2: raise ValueError("alpha_grid must be a 2D array.")
    
    E = [1/np.pi * (coherent(N, alpha) * coherent(N, alpha).dag()) for alpha in alpha_grid.flatten()]
    if numpy:
        return np.array(E)
    else:
        return E

def Husimi_Q_measurement_operators(dim: int, xgrid: np.ndarray, pgrid: np.ndarray) -> np.ndarray:
    """Deprecated alias for heterodyne_measurement_operators."""
    _deprecation_warning('Husimi_Q_measurement_operators', 'heterodyne_measurement_operators')
    return heterodyne_measurement_operators(dim, xgrid=xgrid, pgrid=pgrid)

def homodyne_measurement_operators(N: int, alpha_grid: np.ndarray, numpy: bool=True) -> np.ndarray:
    """
    Computes the measurement operators for homodyne detection of an optical quantum state across a grid of phase space displacements defined by qgrid.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    alpha_grid : np.ndarray
        Complex amplitude grid.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to True.

    Returns
    -------
    np.ndarray
        Measurement operators.
    """
    if not isinstance(N, int): raise ValueError("N must be an integer.")
    if not isinstance(alpha_grid, np.ndarray): raise ValueError("alpha_grid must be a numpy array.")

    E = [1/np.pi * (coherent(N, q) * coherent(N, q).dag()) for q in alpha_grid.flatten()]
    if numpy:
        return np.array(E)
    else:
        return E

def displacement_parity_measurement_operators(N: int, xgrid: np.ndarray=None, pgrid: np.ndarray=None, alpha_grid: np.ndarray=None, numpy: bool=True) -> np.ndarray:
    """
    Computes the measurement operators for displacement parity measurement.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    xgrid : np.ndarray
        Phase space X quadrature grid.
    pgrid : np.ndarray
        Phase space P quadrature grid.
    alpha_grid : np.ndarray
        Complex amplitude grid. If provided, xgrid and pgrid are ignored.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to True.

    Returns
    -------
    np.ndarray
        Measurement operators.
    """
    if not isinstance(N, int): raise ValueError("N must be an integer.")
    if xgrid is None and pgrid is None and alpha_grid is None: raise ValueError("Either xgrid and pgrid, or alpha_grid, must be provided.")

    parity = sum([(-1)**n * fock(N, n) * fock(N, n).dag() for n in range(N)])

    if alpha_grid is None:
        if (xgrid is not None and pgrid is None) or (xgrid is None and pgrid is not None): raise ValueError("Both xgrid and pgrid must be provided together.")
        if not isinstance(xgrid, np.ndarray) or not isinstance(pgrid, np.ndarray): raise ValueError("xgrid and pgrid must be numpy arrays.")
        X, P = np.meshgrid(xgrid, pgrid)
        alpha_grid = X + 1j * P
    else:
        if not isinstance(alpha_grid, np.ndarray): raise ValueError("alpha_grid must be a numpy array.")
        if alpha_grid.ndim != 2: raise ValueError("alpha_grid must be a 2D array.")

    E = []
    for alpha in alpha_grid.flatten():
        D = displace(N, alpha)
        E.append(1/np.pi * (D * parity * D.dag()))
    
    if numpy:
        return np.array(E)
    else:
        return E

def photon_number_measurement_operators(N: int, numpy: bool=True, dim=None) -> np.ndarray:
    """
    Computes the measurement operators for photon occupation number measurement.
    
    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to True.
    
    Returns
    -------
    np.ndarray
        Measurement operators.
    """
    if dim is not None:
        _deprecation_warning('dim', 'N')
        N = dim
    if not isinstance(N, int): raise ValueError("N must be an integer.")

    E = [(fock(N, n) * fock(N, n).dag()) for n in range(N)]
    
    if numpy:
        return np.array(E)
    else:
        return E


### Generalised measurement operators ###

def measurement_operators(N: int, measurement_type: str, numpy: bool=True, dim=None, **kwargs) -> np.ndarray:
    """
    Computes the measurement operators for the specified measurement type.

    Parameters
    ----------
    N : int
        Hilbert space dimensionality.
    measurement_type : str
        Type of measurement to be performed. Must be one of 'heterodyne', 'homodyne', 'displacement_parity', or 'photon_number'.
    numpy : bool
        If True, returns the result as a NumPy array. Defaults to True.
    **kwargs : dict
        Additional keyword arguments required for specific measurement types.

    Returns
    -------
    np.ndarray
        Measurement operators.
    """
    if dim is not None:
        _deprecation_warning('dim', 'N')
        N = dim
    if not isinstance(N, int): raise ValueError("N must be an integer.")

    if measurement_type == 'heterodyne' or measurement_type == 'Husimi_Q' or measurement_type == 'Husimi-Q':
        if measurement_type == 'Husimi-Q':
            _deprecation_warning('Husimi-Q', 'heterodyne')
        if measurement_type == 'Husimi_Q':
            _deprecation_warning('Husimi_Q', 'heterodyne')
        if 'alpha_grid' in kwargs or ('xgrid' in kwargs and 'pgrid' in kwargs):
            return heterodyne_measurement_operators(N, kwargs.get('xgrid'), kwargs.get('pgrid'), kwargs.get('alpha_grid'), numpy=numpy)
        else:
            raise ValueError("For heterodyne detection, either alpha_grid, or both xgrid and pgrid, must be provided.")
    elif measurement_type == 'homodyne':
        if 'alpha_grid' not in kwargs: raise ValueError("For homodyne detection, alpha_grid must be provided.")
        return homodyne_measurement_operators(N, kwargs['alpha_grid'], numpy=numpy)
    elif measurement_type == 'displacement_parity':
        if 'alpha_grid' in kwargs or ('xgrid' in kwargs and 'pgrid' in kwargs):
            return displacement_parity_measurement_operators(N, kwargs.get('xgrid'), kwargs.get('pgrid'), kwargs.get('alpha_grid'), numpy=numpy)
        else:
            raise ValueError("For displacement parity measurement, either alpha_grid, or both xgrid and pgrid, must be provided.")
    elif measurement_type == 'photon_number':
        if 'dim_limit' in kwargs:
            _deprecation_warning('dim_limit', 'N_limit')
            N = kwargs['dim_limit']
        if 'N_limit' in kwargs:
            N = kwargs['N_limit']
        return photon_number_measurement_operators(N, numpy=numpy)
    else:
        raise ValueError(f"Measurement type {measurement_type} not recognized.")
    

##### Single shot measurement results #####

def measure_shots(probabilities: np.ndarray, num_shots: int) -> np.ndarray:
    """
    Simulates a finite number of measurements of a quantum state from a normalized probability distribution.

    Parameters
    ----------
    probabilities : np.ndarray
        Probabilities of the measurement outcomes.
    num_shots : int
        Number of shots to simulate.

    Returns
    -------
    np.ndarray
        Noisy probabilities after applying shot noise.
    """
    if type(probabilities) != np.ndarray: raise ValueError("unrecognized data type for probabilities, expected np.ndarray.")
    if not isinstance(num_shots, (int, float)) or num_shots < 0: raise ValueError("num_shots must be a positive integer.")

    counts = np.random.multinomial(num_shots, probabilities)
    noisy_probs = counts / num_shots
    return noisy_probs.reshape(1, len(counts))  # Reshape to 2D array for modelling