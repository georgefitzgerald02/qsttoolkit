import numpy as np
import scipy.special
import random
from qutip import coherent, Qobj, fock, ket2dm
import warnings

from qsttoolkit.data.num_state_coeffs import states17, statesM, statesP, statesP2, statesM2
from qsttoolkit.utils import _range_error


##### State vectors #####

def cat_state(dim: int, alpha: float, N=None) -> Qobj:
    """
    Generates a cat state (superposition of coherent states).

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    alpha: float
        Coherent state parameter.

    Returns
    -------
    Qobj
        Cat state.
    """
    if not dim:
        if N:
            dim = N
            warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")
    
    if not isinstance(dim, int): raise TypeError("dim must be an integer")

    return (coherent(dim, alpha) + coherent(dim, -alpha)).unit()

def num_state(state: str, dim: int=None, state_index: int=None, N=None) -> Qobj:
    """
    Generates a 'num state' (specific superposition of Fock states numerically optimized for quantum error correction).

    Parameters
    ----
    state: str
        Type of num state to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    dim: int
        Hilbert space dimensionality. If None, the dimension of the state vector will be used.
    state_index: int
        Index of the state to generate. If None, a random state will be generated.

    Returns
    -------
    Qobj
        'Num state'.
    """
    if N:
        dim = N
        warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)

    if not isinstance(dim, int): raise TypeError("dim must be an integer.")

    if state == '17':
        coeffs = states17
    elif state == 'M':
        coeffs = statesM
    elif state == 'P':
        coeffs = statesP
    elif state == 'P2':
        coeffs = statesP2
    elif state == 'M2':
        coeffs = statesM2
    else:
        raise ValueError("state must be one of '17', 'M', 'P', 'P2', or 'M2'")
    
    if state_index is not None:
        if state_index < 0:
            raise ValueError("state_index must be non-negative.")
        elif state_index >= len(coeffs):
            raise ValueError("state_index must be less than the number of states in the given type set.")
    else:
        state_index = random.randint(0, len(coeffs) - 1)

    vector = coeffs[state_index]
    
    if dim is not None:
        if dim < len(vector):
            raise ValueError("dim must be greater than or equal to the length of the state vector.")
        elif dim > len(vector):
            # Extend vector with zeros
            vector = np.append(vector, np.zeros((dim - len(vector), 1)), axis=0)
    
    return Qobj(vector).unit()

def binomial_state(dim: int, S: int, N: int, mu: int, Nc=None) -> Qobj:
    """
    Generates a binomial superposition of Fock states.

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    S: int
        Coherent state parameter.
    N: int
        Number of excitations.
    mu: int
        Logical encoding parameter.
        
    Returns
    -------
    Qobj
        Binomial state in the Nc-dimensional Hilbert space.
    """
    if not dim:
        if Nc:
            dim = Nc
            warnings.warn("Nc is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")

    if not isinstance(dim, int): raise TypeError("dim must be an integer.")

    if S < 1 or S > 10:
        raise ValueError("S must be between 1 and 10.")
    # if N < 2 or N > (dim // (S + 1))-1:                                    Ignored for now, will be reintroduced in a future version
    #     raise ValueError("N must be between 2 and dim/(S+1) - 1")

    return sum([(-1 ** (mu*m)) * np.sqrt(scipy.special.binom(N+1, m)) * fock(dim, (S+1)*m) for m in range(N+1)]).unit()

def gkp_state(dim: int, n1_range: list[int, int], n2_range: list[int, int], delta: float, mu: int, N=None) -> Qobj:
    """
    Generates a Gottesman-Kitaev-Preskill (GKP) state.

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    n1_range: list
        Grid parameter 1.
    n2_range: int
        Grid parameter 2.
    delta: float
        Real normalisation parameter.
    mu: int
        Logical encoding parameter.

    Returns
    -------
    Qobj
        GKP state.
    """
    if not dim:
        if N:
            dim = N
            warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")

    if not isinstance(dim, int): raise TypeError("dim must be an integer.")
    _range_error(n1_range, integers=True, positive=False)
    _range_error(n2_range, integers=True, positive=False)
    # if not isinstance(mu, int): raise TypeError("mu must be an integer.")             Ignored for now, may be reintroduced in a future version

    grid = np.array([[n1, n2] for n1 in range(n1_range[0], n1_range[1]+1) for n2 in range(n2_range[0], n2_range[1]+1)])
    alphas = np.array([np.sqrt(np.pi/2)*((2*n1 + mu) + 1j*n2) for n1, n2 in grid])

    return sum([np.exp(-delta**2 * np.abs(alpha)**2) * np.exp(-1j * alpha.real * alpha.imag) * coherent(dim, alpha) for alpha in alphas]).unit()


##### Density matrices #####

def cat_dm(dim: int, alpha: float, N=None) -> Qobj:
    """
    Generates a density matrix for a cat state (superposition of coherent states).

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    alpha: float
        Coherent state parameter.

    Returns
    -------
    Qobj
        Density matrix for a cat state.
    """
    if not dim:
        if N:
            dim = N
            warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")
        
    return ket2dm(cat_state(dim, alpha))

def num_dm(state: str, dim=None, state_index=None, N=None) -> Qobj:
    """
    Generates a density matrix for a 'num state' (specific superposition of Fock states numerically optimized for quantum error correction).

    Parameters
    ----
    state: str
        Type of 'num state' to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    dim: int
        Hilbert space dimensionality. If None, the dimension of the state vector will be used.
    state_index: int
        Index of the state to generate. If None, a random state will be generated.

    Returns
    -------
    Qobj
        Density matrix for a 'num state'.
    """
    if not dim:
        if N:
            dim = N
            warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")
        
    return ket2dm(num_state(state, dim, state_index))

def binomial_dm(dim: int, S: int, N: int, mu: int, Nc=None) -> Qobj:
    """
    Generates a density matrix for a binomial superposition of Fock states.

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    S: int
        Coherent state parameter.
    N: int
        Number of excitations.
    mu: int
        Logical encoding parameter.
        
    Returns
    -------
    Qobj
        Density matrix for a binomial state in the Nc-dimensional Hilbert space.
    """
    if not dim:
        if Nc:
            dim = Nc
            warnings.warn("Nc is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")
        
    return ket2dm(binomial_state(dim, S, N, mu))

def gkp_dm(dim: int, n1_range: list[int, int], n2_range: list[int, int], delta: float, mu: int, N=None) -> Qobj:
    """
    Generates a density matrix for a Gottesman-Kitaev-Preskill (GKP) state.

    Parameters
    ----
    dim: int
        Hilbert space dimensionality.
    n1_range: list
        Grid parameter 1.
    n2_range: int
        Grid parameter 2.
    delta: float
        Real normalisation parameter.
    mu: int
        Logical encoding parameter.

    Returns
    -------
    Qobj
        Density matrix for a GKP state.
    """
    if not dim:
        if N:
            dim = N
            warnings.warn("N is deprecated and will be removed in a future version. Please use dim instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("dim must be specified.")

    return ket2dm(gkp_state(dim, n1_range, n2_range, delta, mu))