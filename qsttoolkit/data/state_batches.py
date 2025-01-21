import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from qutip import coherent, fock, thermal_dm, rand_dm, ket2dm, qfunc

from qsttoolkit.data.states import cat_state, binomial_state, num_state, gkp_state
from qsttoolkit.utils import _range_error, _random_complex
from qsttoolkit.data.num_state_coeffs import num_type_to_param


class States:
    """
    Skeleton class for a set of quantum states, with methods for computing various representations and measurement functions.

    Attributes
    ----------
    n_states : int
        The number of quantum states.
    N : int
        The Hilbert space dimensionality.
    states : list
        A list to store the quantum states.
    params : list
        A list to store the parameters of the quantum states.

    Methods
    -------
    normalise():
        Normalises each quantum state.
    density_matrices():
        Computes the density matrix for each quantum state.
    gen_Q(xgrid, pgrid):
        Generates the generalised Q-function for each quantum state on a given grid.
    """
    def __init__(self, n_states: int, N: int):
        self.n_states = n_states
        self.N = N
        self.states = []
        self.params = []

    def normalise(self):
        self.states = [state.unit() for state in self.states]

    def density_matrices(self):
        return [ket2dm(state) for state in self.states]
    
    def gen_Q(self, xgrid, pgrid):
        return [qfunc(state, xgrid, pgrid) for state in self.states]
    

class FockStates(States):
    """
    A class to produce batches of Fock states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of Fock states to generate.
    N : int
        The Hilbert space dimensionality.
    n_range : list[int, int]
        The range of photon numbers n.

    Methods
    -------
    __init__(n_states: int, N: int, n_range: list[int, int]):
        Initialises the FockStates object with the given parameters.
    init_states():
        Generates the Fock states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, n_range: list[int, int]):
        super().__init__(n_states, N)
        _range_error(n_range, integers=True)
        if n_range[1] > N:
            raise ValueError(f"max_n ({n_range[1]}) cannot be greater than N ({N})")
        self.n_range = n_range
        self.init_states()

    def init_states(self):
        n_values = np.random.randint(self.n_range[0], self.n_range[1], self.n_states)
        self.states = [fock(self.N, n) for n in n_values]
        self.params = n_values
        self.normalise()

class CoherentStates(States):
    """
    A class to produce batches of coherent states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of coherent states to generate.
    N : int
        The Hilbert space dimensionality.
    alpha_magnitude_range : list[float, float]
        The range of magnitudes for the phase space position parameter alpha.
        
    Methods
    -------
    __init__(n_states: int, N: int, alpha_magnitude_range: list[float, float]):
        Initialises the coherentStates object with the given parameters.
    init_states():
        Generates the coherent states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, alpha_magnitude_range: list[float, float]):
        super().__init__(n_states, N)
        _range_error(alpha_magnitude_range)
        self.alpha_magnitude_range = alpha_magnitude_range
        self.init_states()

    def init_states(self):
        for _ in range(self.n_states):
            alpha = _random_complex(self.alpha_magnitude_range)
            self.states.append(coherent(self.N, alpha))
            self.params.append(alpha)
        self.normalise()

class ThermalStates(States):
    """
    A class to produce batches of thermal states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of thermal states to generate.
    N : int
        The Hilbert space dimensionality.
    nbar_range : list[float, float]
        The range of mean photon numbers nbar.

    Methods
    -------
    __init__(n_states: int, N: int, nbar_range: list[float, float]):
        Initialises the thermalStates object with the given parameters.
    init_states():
        Generates the thermal states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, nbar_range: list[float, float]):
        super().__init__(n_states, N)
        _range_error(nbar_range)
        self.nbar_range = nbar_range
        self.init_states()

    def init_states(self):
        nbar_values = np.random.randint(self.nbar_range[0], self.nbar_range[1], self.n_states)
        self.states = [thermal_dm(self.N, nbar) for nbar in nbar_values]
        self.params = nbar_values
        warnings.warn("thermal states are currently initialised as density matrices. Calling the product of the .density_matrices() method is equivalent to simply calling .states() attribute. This may change in the future.")

    def density_matrices(self):
        return self.states
    
class NumStates(States):
    """
    A class to produce batches of specific bosonic code states numerically optimised for quantum error correction.

    Attributes
    ----------
    n_states : int
        The number of num states to generate.
    N : int
        The Hilbert space dimensionality.
    types : list[str]
        The types of num states to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.

    Methods
    -------
    __init__(n_states: int, N: int, types: list[str]):
        Initialises the numStates object with the given parameters.
    init_states():
        Generates the num states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, types: list[str]):
        super().__init__(n_states, N)
        self.types = types
        self.init_states()

    def init_states(self):
        for _ in range(int(self.n_states)):
            type_choice = random.choice(self.types)
            self.states.append(num_state(type_choice, self.N))
            self.params.append(num_type_to_param[type_choice])
        self.normalise()
            
class BinomialStates(States):
    """
    A class to produce batches of binomial states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of binomial states to generate.
    N : int
        The Hilbert space dimensionality.
    S_range : list[int, int]
        The range of the S parameter.
    mu_range : list[int, int]
        The range of the logical encoding parameter mu.

    Methods
    -------
    __init__(n_states: int, N: int, n_range: list[int, int], p_range: list[float, float]):
        Initialises the binomialStates object with the given parameters.
    init_states():
        Generates the binomial states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, S_range: list[int, int], mu_range: list[int, int]):
        super().__init__(n_states, N)
        _range_error(S_range, integers=True)
        _range_error(mu_range, integers=True)
        self.S_range = S_range
        self.SN_combs = [
            (S, n)
            for S in range(S_range[0], S_range[1] + 1)
            for n in range(2, N // (S + 1))
        ]
        self.mu_range = mu_range
        self.init_states()

    def init_states(self):
        for _ in range(self.n_states):
            S, n = random.choice(self.SN_combs)
            mu = random.randint(self.mu_range[0], self.mu_range[1])
            self.states.append(binomial_state(self.N, S, n, mu))
            self.params.append(S)
        self.normalise()

class CatStates(States):
    """
    A class to produce batches of cat states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of cat states to generate.
    N : int
        The Hilbert space dimensionality.
    alpha_magnitude_range : list[float, float]
        The range of magnitudes for the coherent state parameter alpha.

    Methods
    -------
    __init__(n_states: int, N: int, n_range: list[int, int], p_range: list[float, float]):
        Initialises the catStates object with the given parameters.
    init_states():
        Generates the cat states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, alpha_magnitude_range: list[float, float]):
        super().__init__(n_states, N)
        _range_error(alpha_magnitude_range)
        self.alpha_magnitude_range = alpha_magnitude_range
        self.init_states()

    def init_states(self):
        for _ in range(self.n_states):
            alpha = _random_complex(self.alpha_magnitude_range)
            self.states.append(cat_state(self.N, alpha))
            self.params.append(alpha)
        self.normalise()

class GKPStates(States):
    """
    A class to produce batches of Gottesman-Kitaev-Preskill (GKP) states with randomised parameters within a given range.

    Attributes
    ----------
    n_states : int
        The number of GKP states to generate.
    N : int
        The Hilbert space dimensionality.
    n1_range : list[int, int]
        The range of grid parameter 1.
    n2_range : list[int, int]
        The range of grid parameter 2.
    delta_range : list[float, float]
        The range of the real normalisation parameter delta.
    mu_range : list[int, int]
        The range of the logical encoding parameter mu.

    Methods
    -------
    __init__(n_states: int, N: int, n1_range: list[int, int], n2_range: list[int, int], delta_range: list[float, float], mu_range: list[int, int]):
        Initialises the GKPStates object with the given parameters.
    init_states():
        Generates the GKP states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int, n1_range: list[int, int], n2_range: list[int, int], delta_range: list[float, float], mu_range: list[int, int]):
        super().__init__(n_states, N)
        _range_error(n1_range, integers=True, positive=False)
        _range_error(n2_range, integers=True, positive=False)
        _range_error(delta_range)
        _range_error(mu_range, integers=True)
        self.n1_range = n1_range
        self.n2_range = n2_range
        self.delta_range = delta_range
        self.mu_range = mu_range
        self.init_states()

    def init_states(self):
        for _ in range(self.n_states):
            delta = random.uniform(self.delta_range[0], self.delta_range[1])
            mu = random.randint(self.mu_range[0], self.mu_range[1])
            self.states.append(gkp_state(self.N, self.n1_range, self.n2_range, delta, mu))
            self.params.append(delta)
        self.normalise()

class RandomStates(States):
    """
    A class to produce random states using QuTiP's rand_dm function.

    Attributes
    ----------
    n_states : int
        The number of random states to generate.
    N : int
        The Hilbert space dimensionality.

    Methods
    -------
    __init__(n_states: int, N: int):
        Initialises the RandomStates object with the given parameters.
    init_states():
        Generates the random states based on the given parameters.
    """
    def __init__(self, n_states: int, N: int):
        super().__init__(n_states, N)
        self.init_states()

    def init_states(self):
        self.states = [rand_dm(self.N) for _ in range(self.n_states)]
        self.params = [0 for _ in range(self.n_states)]
        warnings.warn("Random states are currently initialised as density matrices. Calling the product of the .density_matrices() method is equivalent to simply calling .states() attribute. This may change in the future.")

    def density_matrices(self):
        return self.states