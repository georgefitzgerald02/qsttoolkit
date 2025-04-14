import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, coherent, fock
import tensorflow as tf
import warnings

from qsttoolkit.quantum import fidelity
from qsttoolkit.plots import plot_Hinton, plot_Husimi_Q
from qsttoolkit.utils import _L1_regularisation, _threshold_regularisation, _subplot_number, _subplot_figsize


##### Cholesky parametrization functions #####

def parametrize_density_matrix(rho: tf.Tensor) -> tf.Tensor:
    """
    parametrizes the density matrix using the Cholesky decomposition.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix to be parametrized.

    Returns
    -------
    tf.Tensor
        Cholesky decomposition of the density matrix.
    """
    if type(rho) == Qobj:
        rho = rho.full()

    T = tf.linalg.cholesky(rho)  # Lower triangular (batch_size, dim, dim)

    return T

def parameterise_density_matrix(rho: tf.Tensor) -> tf.Tensor:
    """Deprecated alias for parametrize_density_matrix."""
    warnings.warn("parameterise_density_matrix is deprecated and will be removed in a future version. Please use parametrize_density_matrix instead.", DeprecationWarning, stacklevel=2)
    return parametrize_density_matrix(rho)

def reconstruct_density_matrix(params: tf.Tensor, reg: float=1.0e-10, dim=None) -> tf.Tensor:
    """
    Reconstructs the density matrix from the Cholesky decomposition.

    Parameters
    ----------
    params : tf.Tensor
        Cholesky decomposition of the density matrix.

    Returns
    -------
    tf.Tensor
        Reconstructed density matrix.
    """
    if dim is not None:
        warnings.warn("dim is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)

    # Compute density matrix
    rho = tf.matmul(tf.linalg.adjoint(params), params)

    # Regularisation to prevent singular matrices (adding a small identity term)
    dim = tf.shape(rho)[1]
    rho += reg * tf.eye(dim, dtype=tf.complex128)

    # Normalize to ensure trace = 1
    rho /= tf.linalg.trace(rho)

    return rho


##### Loss functions #####

def log_likelihood(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: tf.Tensor, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01, dim=None) -> tf.Tensor:
    """
    Computes the negative log-likelihood of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : tf.Tensor
        Projective measurement operators corresponding to the measurement outcomes.
    L1_reg : float
        L1 regularisation parameter. Defaults to 0.
    thresh_reg : float
        Threshold regularisation parameter. Defaults to 0.
    thresh_reg_threshold : float
        Threshold for the threshold regularisation. Defaults to 0.01.

    Returns
    -------
    tf.Tensor
        negative log-likelihood of the data given the density matrix.
    """
    if dim is not None:
        warnings.warn("dim is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)

    if type(rho) == Qobj:
        rho = rho.full()

    # Compute probabilities: p_k = Tr(P_k * rho) for all projectors
    probabilities = tf.math.real(tf.linalg.trace(tf.matmul(measurement_operators, rho)))

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute log-likelihood
    log_likelihood = tf.reduce_sum(frequency_data * tf.math.log(probabilities))

    return -log_likelihood + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)


##### Define measurement operators #####
### Specific measurement operators - to be removed ###

def Husimi_Q_measurement_operators(dim: int, xgrid: np.array, pgrid: np.array) -> np.array:
    """
    Computes the measurement operators for the Husimi Q function (projectors of all possible coherent operators created from the phase space provided by xgrid and pgrid).
    
    Parameters
    ----------
    dim : int
        Hilbert space dimensionality.
    xgrid : np.array
        Phase space X quadrature grid.
    pgrid : np.array
        Phase space P quadrature grid.

    Returns
    -------
    np.array
        Measurement operators.
    """
    if not isinstance(dim, int): raise ValueError("dim must be an integer.")
    if not isinstance(xgrid, np.ndarray) or not isinstance(pgrid, np.ndarray): raise ValueError("xgrid and pgrid must be numpy arrays.")

    E = []
    for x in xgrid:
        for p in pgrid:
            E.append(np.outer(coherent(dim, x + 1j*p).full(), coherent(dim, x + 1j*p).full().conj().T))
    return np.array(E)

def photon_number_measurement_operators(dim: int) -> np.array:
    """
    Computes the measurement operators for photon occupation number measurement.
    
    Parameters
    ----------
    dim : int
        Hilbert space dimensionality.
    
    Returns
    -------
    np.array
        Measurement operators.
    """
    if not isinstance(dim, int): raise ValueError("dim must be an integer.")

    E = []
    for n in range(dim):
        E.append(np.outer(fock(dim, n).full(), fock(dim, n).full().conj().T))
    return np.array(E)


### Generalised measurement operators ###

def measurement_operators(dim: int, measurement_type: str, **kwargs) -> np.array:
    """
    Computes the measurement operators for the specified measurement type.

    Parameters
    ----------
    dim : int
        Hilbert space dimensionality.
    measurement_type : str
        Type of measurement to be performed.
    **kwargs : dict
        Additional keyword arguments required for specific measurement types.

    Returns
    -------
    np.array
        Measurement operators.
    """
    if not isinstance(dim, int): raise ValueError("dim must be an integer.")

    E = []
    if measurement_type == 'Husimi_Q' or measurement_type == 'Husimi-Q':
        if measurement_type == 'Husimi-Q':
            warnings.warn("'Husimi-Q' keyword is deprecated and will be removed in a future version. Please use 'Husimi_Q' instead.", DeprecationWarning, stacklevel=2)
        if 'xgrid' not in kwargs or 'pgrid' not in kwargs:
            raise ValueError("For Husimi Q measurement, xgrid and pgrid must be provided.")
        for x in kwargs['xgrid']:
            for p in kwargs['pgrid']:
                E.append(np.outer(coherent(dim, x + 1j*p).full(), coherent(dim, x + 1j*p).full().conj().T))
    elif measurement_type == 'photon_number':
        if 'dim_limit' in kwargs:
            dim = kwargs['dim_limit']
        for n in range(dim):
            E.append(np.outer(fock(dim, n).full(), fock(dim, n).full().conj().T))
    else:
        raise ValueError(f"Measurement type {measurement_type} not recognized.")
    return np.array(E)


##### Define constraints - no longer used by MLE #####

def trace_constraint(params: np.array) -> float:
    """
    Constraint function to ensure the trace of the density matrix is 1.
    
    Parameters
    ----------
    params : np.array
        Flattened vector of real parameters.

    Returns
    -------
    float
        Difference between the trace of the reconstructed density matrix and 1.
    """
    warnings.warn("The trace_constraint function is deprecated and will be removed in a future version. The trace of the density matrix is now enforced by reconstruct_density_matrix function.", DeprecationWarning, stacklevel=2)

    rho = reconstruct_density_matrix(params)
    return np.trace(rho).real - 1  # Should be zero

def positivity_constraint(params: np.array) -> float:
    """
    Constraint to ensure the density matrix is positive semi-definite.
    
    Parameters
    ----------
    params : np.array
        Flattened vector of real parameters.

    Returns
    -------
    float
        Smallest eigenvalue of the reconstructed density matrix.
    """
    warnings.warn("The positivity_constraint function is deprecated and will be removed in a future version. The trace of the density matrix is now enforced by reconstruct_density_matrix function.", DeprecationWarning, stacklevel=2)
    
    rho = reconstruct_density_matrix(params)
    eigenvalues = np.linalg.eigvalsh(rho)  # Eigenvalues of rho
    return np.min(eigenvalues)  # Should be >= 0


##### Parent class for all QST methods #####

class QuantumStateTomography:
    """A parent class for all quantum state tomography methods."""

    def __init__(self):
        self.reconstructed_dm = None
        self.progress_saves = None
        self.fidelities = None
        self.times = None

    def fidelity(self, true_dm: np.ndarray) -> float:
        """
        Computes the fidelity between the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            True density matrix.

        Returns
        -------
        float
            Fidelity between the true and reconstructed density matrices.
        """
        if len(self.reconstructed_dm.shape) != 2:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        return fidelity(true_dm, self.reconstructed_dm)

    def plot_losses(self):
        """Plots the losses over epochs."""
        plt.figure(figsize=(5, 4))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses over epochs')
        plt.show()

    def plot_fidelities(self, true_dm=None):
        """Plots the fidelity between the true and reconstructed density matrices over epochs."""
        if true_dm is not None:
            warnings.warn("true_dm is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)
        
        plt.figure(figsize=(5, 4))
        plt.plot(self.fidelities)
        plt.ylim(0,1)
        plt.xlabel('Epoch')
        plt.ylabel('Fidelity')
        plt.title('Fidelity over epochs')
        plt.show()

    def plot_times(self):
        """Plots the cumulative time taken for each epoch."""
        plt.figure(figsize=(5, 4))
        plt.plot(self.times)
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Time taken after epochs')
        plt.show()

    def plot_comparison_Hintons(self, true_dm: np.ndarray):
        """
        Plots Hinton diagrams of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            True density matrix.
        """
        if type(true_dm) == Qobj:
            true_dm = true_dm.full()
        elif type(true_dm) == tf.Tensor:
            true_dm = true_dm.numpy()
        elif type(true_dm) != np.ndarray:
            raise ValueError("unrecognized data type for true_dm.")

        _, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_Hinton(true_dm, ax=axs[0], label='true density matrix')
        if len(self.reconstructed_dm.shape) == 2:
            reconstruction = self.reconstructed_dm
        else:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        plot_Hinton(reconstruction, ax=axs[1], label='optimized density matrix')
        plt.show()

    def plot_comparison_hintons(self, true_dm: np.ndarray):
        """Deprecated alias for plot_comparison_Hintons."""
        warnings.warn("plot_comparison_hintons is deprecated and will be removed in a future version. Please use plot_comparison_Hintons instead.", DeprecationWarning, stacklevel=2)
        return self.plot_comparison_Hintons(true_dm)

    def plot_comparison_Husimi_Qs(self, true_dm: np.ndarray, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Husimi Q functions of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            True density matrix.
        xgrid : np.ndarray
            Phase space X quadrature grid.
        pgrid : np.ndarray
            Phase space P quadrature grid.
        """
        if not isinstance(xgrid, np.ndarray) or not isinstance(pgrid, np.ndarray): raise ValueError("xgrid and pgrid must be numpy arrays.")
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_Husimi_Q(true_dm, xgrid, pgrid, fig=fig, ax=axs[0], label='true density matrix')
        if len(self.reconstructed_dm.shape) == 2:
            reconstruction = self.reconstructed_dm
        else:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        plot_Husimi_Q(reconstruction, xgrid, pgrid, fig=fig, ax=axs[1], label='reconstructed density matrix')
        plt.show()

    def plot_intermediate_Hintons(self):
        """Plots Hinton diagrams of the density matrices in the progress_saves attribute."""
        if len(self.progress_saves[0].shape) == 2:
            reconstructions = self.progress_saves
        else:
            raise ValueError("Invalid shape of reconstructed density matrices.")

        subplot_number = _subplot_number(len(reconstructions))
        _, axs = plt.subplots(subplot_number[0], subplot_number[1], figsize=_subplot_figsize(len(reconstructions)), squeeze=False)
        axs = np.array(axs).flatten()
        for i, dm in enumerate(reconstructions):
            plot_Hinton(dm, ax=axs[i], label=f"save {i}")
        plt.show()

    def plot_intermediate_hintons(self):
        """Deprecated alias for plot_intermediate_Hintons."""
        warnings.warn("plot_intermediate_hintons is deprecated and will be removed in a future version. Please use plot_intermediate_Hintons instead.", DeprecationWarning, stacklevel=2)
        return self.plot_intermediate_Hintons()

    def plot_intermediate_Husimi_Qs(self, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Husimi Q functions of the density matrices in the progress_saves attribute.

        Parameters
        ----------
        xgrid : np.ndarray
            Phase space X quadrature grid.
        pgrid : np.ndarray
            Phase space P quadrature grid.
        """
        if not isinstance(xgrid, np.ndarray) or not isinstance(pgrid, np.ndarray): raise ValueError("xgrid and pgrid must be numpy arrays.")
        
        if len(self.progress_saves[0].shape) == 2:
            reconstructions = self.progress_saves
        else:
            raise ValueError("Invalid shape of reconstructed density matrices.")
        
        subplot_number = _subplot_number(len(reconstructions))
        fig, axs = plt.subplots(subplot_number[0], subplot_number[1], figsize=_subplot_figsize(len(reconstructions)))
        axs = axs.flatten()
        for i, dm in enumerate(reconstructions):
            plot_Husimi_Q(dm, xgrid, pgrid, fig=fig, ax=axs[i], label=f"save {i}")
        plt.show()