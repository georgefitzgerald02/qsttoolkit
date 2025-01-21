import random
import numpy as np
import scipy.special
from qutip import coherent, Qobj, fock

from qsttoolkit.data.num_state_coeffs import states17, statesM, statesP, statesP2, statesM2


def cat_state(N: int, alpha: float) -> Qobj:
    """
    Generates a cat state (superposition of coherent states) in the N-dimensional Hilbert space.

    Parameters
    ----
    N: int
        Dimension of the Hilbert space.
    alpha: float
        Coherent state parameter.

    Returns
    -------
    Qobj
        Cat state in the N-dimensional Hilbert space.
    """
    return (coherent(N, alpha) + coherent(N, -alpha)).unit()

def num_state(state: str, N=None, state_index=None) -> Qobj:
    """
    Generates a 'num state' (specific superposition of Fock states numerically optimised for quantum error correction) in the N-dimensional Hilbert space.

    Parameters
    ----
    state: str
        Type of 'num state' to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    N: int, optional
        Dimension of the Hilbert space. If None, the dimension of the state vector will be used.
    state_index: int, optional
        Index of the state to generate. If None, a random state will be generated.

    Returns
    -------
    Qobj
        'Num state' in the N-dimensional Hilbert space.
    """
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
            raise ValueError("state_index must be non-negative")
        elif state_index >= len(coeffs):
            raise ValueError("state_index must be less than the number of states in the given type set")
    else:
        state_index = random.randint(0, len(coeffs) - 1)

    vector = coeffs[state_index]
    
    if N is not None:
        if N < len(vector):
            raise ValueError("N must be greater than or equal to the length of the state vector")
        elif N > len(vector):
            # Extend vector with zeros
            vector = np.append(vector, np.zeros((N - len(vector), 1)), axis=0)
    
    return Qobj(vector).unit()

def binomial_state(Nc: int, S: int, N: int, mu: int) -> Qobj:
    """
    Generates a binomial superposition of Fock states in the N-dimensional Hilbert space.

    Parameters
    ----
    Nc: int
        Dimension of the Hilbert space.
    S: int
        Coherent state parameter.
    N: int
        Number of excitations.
    mu: int
        Logical encoding.
        
    Returns
    -------
    Qobj
        Binomial state in the Nc-dimensional Hilbert space.
    """
    if S < 1 or S > 10:
        raise ValueError("S must be between 1 and 10")
    if N < 2 or N > (Nc // (S + 1))-1:
        raise ValueError("N must be between 2 and Nc/(S+1) - 1")

    return sum([(-1 ** (mu*m)) * np.sqrt(scipy.special.binom(N+1, m)) * fock(Nc, (S+1)*m) for m in range(N+1)]).unit()

def gkp_state(N: int, n1_range: list[int, int], n2_range: list[int, int], delta: float, mu: int) -> Qobj:
    """
    Generates a Gottesman-Kitaev-Preskill (GKP) state in the N-dimensional Hilbert space.

    Parameters
    ----
    N: int
        Dimension of the Hilbert space.
    n1_range: list
        Grid parameter 1.
    n2_range: int
        Grid parameter 2.
    delta: float
        Real normalisation parameter.
    mu: int
        Logical encoding.

    Returns
    -------
    Qobj
        GKP state in the N-dimensional Hilbert space.
    """
    grid = np.array([[n1, n2] for n1 in range(n1_range[0], n1_range[1]+1) for n2 in range(n2_range[0], n2_range[1]+1)])
    alphas = np.array([np.sqrt(np.pi/2)*((2*n1 + mu) + 1j*n2) for n1, n2 in grid])

    return sum([np.exp(-delta**2 * np.abs(alpha)**2) * np.exp(-1j * alpha.real * alpha.imag) * coherent(N, alpha) for alpha in alphas]).unit()