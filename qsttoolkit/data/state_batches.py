import numpy as np
import random
from qutip import Qobj, coherent, fock, thermal_dm, rand_dm, ket2dm, qfunc
import warnings

from qsttoolkit.data.states import cat_state, binomial_state, num_state, gkp_state
from qsttoolkit.data.num_state_coeffs import num_type_to_param
from qsttoolkit.utils import _range_error, _random_complex, _deprecation_warning


class States:
    """
    Skeleton class for a set of quantum states, with methods for computing various representations and measurement functions.

    Attributes
    ----------
    n_states : int
        Number of states to generate.
    dim : int
        Hilbert space dimensionality.
    """
    def __init__(self, n_states: int, dim: int):
        """Initializes the States object with the given parameters."""
        if not isinstance(dim, int): raise ValueError("dim must be an integer.")
        if not isinstance(n_states, int): raise ValueError("n_states must be an integer.")

        self.n_states = n_states
        self.dim = dim
        self.states = []
        self.params = []

    def normalize(self):
        """Normalizes the states within the self.states attribute."""
        self.states = [state.unit() for state in self.states]

    def normalise(self):
        """Deprecated alias for the normalize method."""
        _deprecation_warning('normalise', 'normalize')
        self.normalize()

    def density_matrices(self) -> list[Qobj]:
        """
        Returns the density matrices.

        Returns
        -------
        list[Qobj]
            Density matrices of the states.
        """ 
        return [ket2dm(state) for state in self.states]
    
    def gen_Q(self, xgrid: np.ndarray, pgrid: np.ndarray) -> list[np.ndarray]:
        """
        Generates the Husimi Q function images for the states.
        
        Parameters
        ----------
        xgrid : np.ndarray
            Grid for the real part of the coherent state parameter.
        pgrid : np.ndarray
            Grid for the imaginary part of the coherent state parameter.

        Returns
        -------
        list[np.ndarray]
            Husimi Q function images for the states.
        """
        return [qfunc(state, xgrid, pgrid) for state in self.states]
    

class FockStates(States):
    """
    A class to produce batches of Fock states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of Fock states to generate.
    dim : int
        Hilbert space dimensionality.
    n_range : list[int, int]
        Range of photon numbers n.
    """
    def __init__(self, n_states: int, dim: int, n_range: list[int, int], N=None):
        """Initializes the FockStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N

        super().__init__(n_states, dim)
        _range_error(n_range, integers=True)
        if n_range[1] > dim: raise ValueError(f"max_n ({n_range[1]}) cannot be greater than dim ({dim})")
        self.n_range = n_range
        self.init_states()

    def init_states(self):
        """Generates the Fock states with the given parameters."""
        n_values = np.random.randint(self.n_range[0], self.n_range[1], self.n_states)
        self.states = [fock(self.dim, n) for n in n_values]
        self.params = n_values
        self.normalize()

class CoherentStates(States):
    """
    A class to produce batches of coherent states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of coherent states to generate.
    dim : int
        Hilbert space dimensionality.
    alpha_magnitude_range : list[float, float]
        Range of magnitudes for the phase space position parameter alpha.
    """
    def __init__(self, n_states: int, dim: int, alpha_magnitude_range: list[float, float], N=None):
        """Initializes the CoherentStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N

        super().__init__(n_states, dim)
        _range_error(alpha_magnitude_range)
        self.alpha_magnitude_range = alpha_magnitude_range
        self.init_states()

    def init_states(self):
        """Generates the coherent states with the given parameters."""
        for _ in range(self.n_states):
            alpha = _random_complex(self.alpha_magnitude_range)
            self.states.append(coherent(self.dim, alpha))
            self.params.append(alpha)
        self.normalize()

class ThermalStates(States):
    """
    A class to produce batches of thermal states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of thermal states to generate.
    dim : int
        Hilbert space dimensionality.
    nbar_range : list[float, float]
        Range of mean photon numbers nbar.
    """
    def __init__(self, n_states: int, dim: int, nbar_range: list[float, float], N=None):
        """Initializes the ThermalStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N

        super().__init__(n_states, dim)
        _range_error(nbar_range)
        self.nbar_range = nbar_range
        self.init_states()

    def init_states(self):
        """Generates the thermal states with the given parameters."""
        nbar_values = np.random.randint(self.nbar_range[0], self.nbar_range[1], self.n_states)
        self.states = [thermal_dm(self.dim, nbar) for nbar in nbar_values]
        self.params = nbar_values
        warnings.warn("thermal states are currently initialized as density matrices. Calling the product of the .density_matrices() method is equivalent to simply calling .states() attribute. This may change in the future.")

    def density_matrices(self):
        """Overrides the parent method .density_matrices() as thermal states are initialized as density matrices."""
        return self.states
    
class NumStates(States):
    """
    A class to produce batches of specific bosonic code states numerically optimized for quantum error correction.

    Attributes
    ----------
    n_states : int
        Number of num states to generate.
    dim : int
        Hilbert space dimensionality.
    types : list[str]
        Types of num states to generate. Must be one of '17', 'M', 'P', 'P2', or 'M2'.
    """
    def __init__(self, n_states: int, dim: int, types: list[str], N=None):
        """Initializes the NumStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N
        
        super().__init__(n_states, dim)
        self.types = types
        self.init_states()

    def init_states(self):
        """Generates the num states with the given parameters."""
        for _ in range(int(self.n_states)):
            type_choice = random.choice(self.types)
            self.states.append(num_state(type_choice, self.dim))
            self.params.append(num_type_to_param[type_choice])
        self.normalize()
            
class BinomialStates(States):
    """
    A class to produce batches of binomial states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of binomial states to generate.
    dim : int
        Hilbert space dimensionality.
    S_range : list[int, int]
        Range of the S parameter.
    mu_range : list[int, int]
        Range of the logical encoding parameter mu.
    """
    def __init__(self, n_states: int, dim: int, S_range: list[int, int], mu_range: list[int, int], N=None):
        """Initializes the BinomialStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N

        super().__init__(n_states, dim)
        _range_error(S_range, integers=True)
        _range_error(mu_range, integers=True)
        self.S_range = S_range
        self.SN_combs = [
            (S, N)
            for S in range(S_range[0], S_range[1] + 1)
            for N in range(2, dim // (S + 1))
        ]
        self.mu_range = mu_range
        self.init_states()

    def init_states(self):
        """Generates the binomial states with the given parameters."""
        for _ in range(self.n_states):
            S, N = random.choice(self.SN_combs)
            mu = random.randint(self.mu_range[0], self.mu_range[1])
            self.states.append(binomial_state(self.dim, S, N, mu))
            self.params.append(S)
        self.normalize()

class CatStates(States):
    """
    A class to produce batches of cat states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of cat states to generate.
    dim : int
        Hilbert space dimensionality.
    alpha_magnitude_range : list[float, float]
        Range of magnitudes for the coherent state parameter alpha.
    type : str
        Type of cat state to generate. Must be one of 'positive', 'negative', or 'mix'.
    """
    def __init__(self, n_states: int, dim: int, alpha_magnitude_range: list[float, float], type: str='positive', N=None):
        """Initializes the CatStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N
        
        super().__init__(n_states, dim)
        _range_error(alpha_magnitude_range)
        if type not in ['positive', 'negative', 'mix']: raise ValueError("type must be either 'positive', 'negative' or 'mix'.")
        self.alpha_magnitude_range = alpha_magnitude_range
        self.type = type
        self.init_states()

    def init_states(self):
        """Generates the cat states with the given parameters."""
        for _ in range(self.n_states):
            alpha = _random_complex(self.alpha_magnitude_range)
            if self.type == 'positive':
                self.states.append(cat_state(self.dim, alpha, positive=True))
            elif self.type == 'negative':
                self.states.append(cat_state(self.dim, alpha, positive=False))
            elif self.type == 'mix':
                if random.random() < 0.5:
                    self.states.append(cat_state(self.dim, alpha, positive=True))
                else:
                    self.states.append(cat_state(self.dim, alpha, positive=False))
            self.params.append(alpha)
        self.normalize()

class GKPStates(States):
    """
    A class to produce batches of Gottesman-Kitaev-Preskill (GKP) states with randomized parameters within a given range.

    Attributes
    ----------
    n_states : int
        Number of GKP states to generate.
    dim : int
        Hilbert space dimensionality.
    n1_range : list[int, int]
        Range of grid parameter 1.
    n2_range : list[int, int]
        Range of grid parameter 2.
    delta_range : list[float, float]
        Range of the real normalisation parameter delta.
    mu_range : list[int, int]
        Range of the logical encoding parameter mu.
    """
    def __init__(self, n_states: int, dim: int, n1_range: list[int, int], n2_range: list[int, int], delta_range: list[float, float], mu_range: list[int, int], N=None):
        """Initializes the GKPStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N
        
        super().__init__(n_states, dim)
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
        """Generates the GKP states with the given parameters."""
        for _ in range(self.n_states):
            delta = random.uniform(self.delta_range[0], self.delta_range[1])
            mu = random.randint(self.mu_range[0], self.mu_range[1])
            self.states.append(gkp_state(self.dim, self.n1_range, self.n2_range, delta, mu))
            self.params.append(delta)
        self.normalize()

class RandomStates(States):
    """
    A class to produce random states using QuTiP's rand_dm function.

    Attributes
    ----------
    n_states : int
        Number of random states to generate.
    dim : int
        Hilbert space dimensionality.
    """
    def __init__(self, n_states: int, dim: int, N=None):
        """Initializes the RandomStates object with the given parameters."""
        if N:
            _deprecation_warning('N', 'dim')
            dim = N
        
        super().__init__(n_states, dim)
        self.init_states()

    def init_states(self):
        """Generates the random states with the given parameters"""
        self.states = [rand_dm(self.dim) for _ in range(self.n_states)]
        self.params = [0 for _ in range(self.n_states)]
        warnings.warn("Random states are currently initialized as density matrices. Calling the product of the .density_matrices() method is equivalent to simply calling .states() attribute. This may change in the future.")

    def density_matrices(self):
        """Overrides the parent method .density_matrices() as random states are initialized as density matrices."""
        return self.states