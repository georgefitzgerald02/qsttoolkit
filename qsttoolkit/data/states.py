import numpy as np
import scipy.special
import random
from qutip import Qobj, tensor, basis, ket2dm, bell_state, ghz_state, coherent, fock

from qsttoolkit.data.num_state_coeffs import states17, statesM, statesP, statesP2, statesM2
from qsttoolkit.quantum import hadamard
from qsttoolkit.utils import _range_error, _deprecation_warning


##### Discrete-variable quantum states #####
### State vectors ###

def hadamard_state(N: int) -> Qobj:
    """
    Generates an N-qubit Hadamard state |Φ_H⟩ = (|0⟩ + |1⟩)/√2 ⊗ ... ⊗ (|0⟩ + |1⟩)/√2, equivalent to applying a Hadamard gate to every qubit in |0⟩^⊗N.

    Parameters
    ----------
    N : int
        Number of qubits.
    
    Returns
    -------
    Qobj
        Hadamard state.
    """
    if N < 1: raise ValueError("Number of qubits must be at least 1.")

    zero_state = tensor([basis(2, 0)] * N)
    H_all = tensor([hadamard()] * N)
    return H_all * zero_state

### Density matrices ###

def hadamard_dm(N: int) -> Qobj:
    """
    Generates a density matrix for the N-qubit Hadamard state |Φ_H⟩ = (|0⟩ + |1⟩)/√2 ⊗ ... ⊗ (|0⟩ + |1⟩)/√2, equivalent to applying a Hadamard gate to every qubit in |0⟩^⊗N.

    Parameters
    ----------
    N : int
        Number of qubits.
    
    Returns
    -------
    Qobj
        Density matrix for the Hadamard state.
    """
    if N < 1: raise ValueError("Number of qubits must be at least 1.")

    return ket2dm(hadamard_state(N))

def bell_dm(state: str='00') -> Qobj:
    """
    Generates a density matrix for a given Bell state.

    Parameters
    ----------
    state : str
        Type of Bell state to generate. Must be one of '00', '01', '10', or '11'.
    
    Returns
    -------
    Qobj
        Density matrix for the Bell state.
    """
    return ket2dm(bell_state(state))

def ghz_dm(N: int) -> Qobj:
    """
    Generates a density matrix for an N-qubit Greenberger-Horne-Zeilinger (GHZ) state.

    Parameters
    ----------
    N : int
        Number of qubits.

    Returns
    -------
    Qobj
        Density matrix for the GHZ state.
    """
    return ket2dm(ghz_state(N))

##### Continuous-variable (optical) quantum states #####
### State vectors ###

def cat_state(N: int, alpha: float, positive: bool=True, dim=None) -> Qobj:
    """
    Generates a cat state (superposition of coherent states).

    Parameters
    ----
    N : int
        Hilbert space dimensionality.
    alpha : float
        Coherent state parameter.
    positive : bool
        If True, generates a positive cat state (superposition of coherent states with the same phase). If False, generates a negative cat state (superposition of coherent states with opposite phases).

    Returns
    -------
    Qobj
        Cat state.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim
    
    if not isinstance(N, int): raise TypeError("N must be an integer")
    if not isinstance(positive, bool): raise TypeError("positive must be a boolean")

    if positive:
        return (coherent(N, alpha) + coherent(N, -alpha)).unit()
    else:
        return (coherent(N, alpha) - coherent(N, -alpha)).unit()

def num_state(state: str, N: int=None, state_index: int=None, dim=None) -> Qobj:
    """
    Generates a 'num state' (specific superposition of Fock states numerically optimized for quantum error correction).

    Parameters
    ----
    state : str
        Type of num state to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    N : int
        Hilbert space dimensionality. If None, the dimension of the state vector will be used.
    state_index : int
        Index of the state to generate. If None, a random state will be generated.

    Returns
    -------
    Qobj
        'Num state'.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim

    if not isinstance(N, int): raise TypeError("N must be an integer.")

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
    
    if N is not None:
        if N < len(vector):
            raise ValueError("N must be greater than or equal to the length of the state vector.")
        elif N > len(vector):
            # Extend vector with zeros
            vector = np.append(vector, np.zeros((N - len(vector), 1)), axis=0)
    
    return Qobj(vector).unit()

def binomial_state(dim: int, S: int, N: int, mu: int, Nc=None) -> Qobj:
    """
    Generates a binomial superposition of Fock states.

    Parameters
    ----
    dim : int
        Hilbert space dimensionality.
    S : int
        Coherent state parameter.
    N : int
        Number of excitations.
    mu : int
        Logical encoding parameter.
        
    Returns
    -------
    Qobj
        Binomial state in the Nc-dimensional Hilbert space.
    """
    if not dim:
        if Nc:
            dim = Nc
            _deprecation_warning('Nc', 'dim')
        else:
            raise ValueError("dim must be specified.")

    if not isinstance(dim, int): raise TypeError("dim must be an integer.")

    if S < 1 or S > 10:
        raise ValueError("S must be between 1 and 10.")
    # if N < 2 or N > (dim // (S + 1))-1:                                    Ignored for now, will be reintroduced in a future version
    #     raise ValueError("N must be between 2 and dim/(S+1) - 1")

    return sum([(-1 ** (mu*m)) * np.sqrt(scipy.special.binom(N+1, m)) * fock(dim, (S+1)*m) for m in range(N+1)]).unit()

def gkp_state(N: int, n1_range: list[int, int], n2_range: list[int, int], delta: float, mu: int, dim=None) -> Qobj:
    """
    Generates a Gottesman-Kitaev-Preskill (GKP) state.

    Parameters
    ----
    N : int
        Hilbert space dimensionality.
    n1_range : list
        Grid parameter 1.
    n2_range : int
        Grid parameter 2.
    delta : float
        Real normalisation parameter.
    mu : int
        Logical encoding parameter.

    Returns
    -------
    Qobj
        GKP state.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim

    if not isinstance(N, int): raise TypeError("N must be an integer.")
    _range_error(n1_range, integers=True, positive=False)
    _range_error(n2_range, integers=True, positive=False)
    # if not isinstance(mu, int): raise TypeError("mu must be an integer.")             Ignored for now, may be reintroduced in a future version
    
    grid = np.mgrid[n1_range[0]:n1_range[1]+1, n2_range[0]:n2_range[1]+1].reshape(2, -1)
    alphas = np.sqrt(np.pi/2) * ((2 * grid[0] + mu) + 1j * grid[1])
    weights = np.exp(-delta**2 * np.abs(alphas)**2 - 1j * alphas.real * alphas.imag)

    return sum(weights[i] * coherent(N, alphas[i]) for i in range(alphas.size)).unit()

### Density matrices ###

def cat_dm(N: int, alpha: float, positive: bool=True, dim=None) -> Qobj:
    """
    Generates a density matrix for a cat state (superposition of coherent states).

    Parameters
    ----
    N : int
        Hilbert space dimensionality.
    alpha : float
        Coherent state parameter.
    positive : bool
        If True, generates a positive cat state (superposition of coherent states with the same phase). If False, generates a negative cat state (superposition of coherent states with opposite phases).

    Returns
    -------
    Qobj
        Density matrix for a cat state.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim
        
    return ket2dm(cat_state(N, alpha, positive))

def num_dm(state: str, N=None, state_index=None, dim=None) -> Qobj:
    """
    Generates a density matrix for a 'num state' (specific superposition of Fock states numerically optimized for quantum error correction).

    Parameters
    ----
    state : str
        Type of 'num state' to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    N : int
        Hilbert space dimensionality. If None, the dimension of the state vector will be used.
    state_index : int
        Index of the state to generate. If None, a random state will be generated.

    Returns
    -------
    Qobj
        Density matrix for a 'num state'.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim

    return ket2dm(num_state(state, N, state_index))

def binomial_dm(dim: int, S: int, N: int, mu: int, Nc=None) -> Qobj:
    """
    Generates a density matrix for a binomial superposition of Fock states.

    Parameters
    ----
    dim : int
        Hilbert space dimensionality.
    S : int
        Coherent state parameter.
    N : int
        Number of excitations.
    mu : int
        Logical encoding parameter.
        
    Returns
    -------
    Qobj
        Density matrix for a binomial state in the Nc-dimensional Hilbert space.
    """
    if Nc:
        _deprecation_warning('Nc', 'dim')
        dim = Nc
  
    return ket2dm(binomial_state(dim, S, N, mu))

def gkp_dm(N: int, n1_range: list[int, int], n2_range: list[int, int], delta: float, mu: int, dim=None) -> Qobj:
    """
    Generates a density matrix for a Gottesman-Kitaev-Preskill (GKP) state.

    Parameters
    ----
    N : int
        Hilbert space dimensionality.
    n1_range : list
        Grid parameter 1.
    n2_range : int
        Grid parameter 2.
    delta : float
        Real normalisation parameter.
    mu : int
        Logical encoding parameter.

    Returns
    -------
    Qobj
        Density matrix for a GKP state.
    """
    if dim:
        _deprecation_warning('dim', 'N')
        N = dim

    return ket2dm(gkp_state(N, n1_range, n2_range, delta, mu))