import numpy as np
import matplotlib.pyplot as plt
import time
from qutip import Qobj
import tensorflow as tf
import warnings

from qsttoolkit.quantum import fidelity
from qsttoolkit.plots import plot_hinton, plot_husimi_Q, plot_wigner
from qsttoolkit.utils import _subplot_number, _subplot_figsize, _deprecation_warning, _no_longer_required_warning
from qsttoolkit.data.measurement import Husimi_Q_measurement_operators, photon_number_measurement_operators, measurement_operators
from qsttoolkit.tomography.loss import log_likelihood


##### Cholesky parametrization functions - once more are introduced, move to their own file #####

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
    if type(rho) == Qobj: rho = rho.full()

    T = tf.linalg.cholesky(rho)  # Lower triangular (batch_size, dim, dim)

    return T

def parameterise_density_matrix(rho: tf.Tensor) -> tf.Tensor:
    """Deprecated alias for parametrize_density_matrix."""
    _deprecation_warning('parameterise_density_matrix', 'parametrize_density_matrix')
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
    if dim is not None: _no_longer_required_warning('dim')

    # Compute density matrix
    rho = tf.matmul(tf.linalg.adjoint(params), params)

    # Regularisation to prevent singular matrices (adding a small identity term)
    dim = tf.shape(rho)[1]
    rho += reg * tf.eye(dim, dtype=tf.complex128)

    # Normalize to ensure trace = 1
    rho /= tf.linalg.trace(rho)

    return rho


##### Define constraints - no longer used by MLE #####

def trace_constraint(params: np.ndarray) -> float:
    """
    Constraint function to ensure the trace of the density matrix is 1.
    
    Parameters
    ----------
    params : np.ndarray
        Flattened vector of real parameters.

    Returns
    -------
    float
        Difference between the trace of the reconstructed density matrix and 1.
    """
    warnings.warn("The trace_constraint function is deprecated and will be removed in a future version. The trace of the density matrix is now enforced by reconstruct_density_matrix function.", DeprecationWarning, stacklevel=2)

    rho = reconstruct_density_matrix(params)
    return np.trace(rho).real - 1  # Should be zero

def positivity_constraint(params: np.ndarray) -> float:
    """
    Constraint to ensure the density matrix is positive semi-definite.
    
    Parameters
    ----------
    params : np.ndarray
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
        if len(self.reconstructed_dm.shape) != 2: raise ValueError("Invalid shape of reconstructed density matrix.")
        
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
        if true_dm is not None: _no_longer_required_warning('true_dm')
        
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

    def plot_comparison_hintons(self, true_dm: np.ndarray):
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
        plot_hinton(true_dm, ax=axs[0], label='true density matrix')
        if len(self.reconstructed_dm.shape) == 2:
            reconstruction = self.reconstructed_dm
        else:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        plot_hinton(reconstruction, ax=axs[1], label='optimized density matrix')
        plt.show()

    def plot_comparison_Hintons(self, true_dm: np.ndarray):
        """Deprecated alias for plot_comparison_hintons. Plots Hinton diagrams of the true and reconstructed density matrices."""
        _deprecation_warning('plot_comparison_Hintons', 'plot_comparison_hintons')
        return self.plot_comparison_hintons(true_dm)

    def plot_comparison_husimi_Qs(self, true_dm: np.ndarray, xgrid: np.ndarray, pgrid: np.ndarray):
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
        plot_husimi_Q(true_dm, xgrid, pgrid, fig=fig, ax=axs[0], label='true density matrix')
        if len(self.reconstructed_dm.shape) == 2:
            reconstruction = self.reconstructed_dm
        else:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        plot_husimi_Q(reconstruction, xgrid, pgrid, fig=fig, ax=axs[1], label='reconstructed density matrix')
        plt.show()

    def plot_comparison_Husimi_Qs(self, true_dm: np.ndarray, xgrid: np.ndarray, pgrid: np.ndarray):
        """Deprecated alias for plot_comparison_husimi_Qs. Plots the Husimi Q functions of the true and reconstructed density matrices."""
        _deprecation_warning('plot_comparison_Husimi_Qs', 'plot_comparison_husimi_Qs')
        return self.plot_comparison_husimi_Qs(true_dm, xgrid, pgrid)
    
    def plot_comparison_wigners(self, true_dm: np.ndarray, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Wigner functions of the true and reconstructed density matrices.

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
        plot_wigner(true_dm, xgrid, pgrid, fig=fig, ax=axs[0], label='true density matrix')
        if len(self.reconstructed_dm.shape) == 2:
            reconstruction = self.reconstructed_dm
        else:
            raise ValueError("Invalid shape of reconstructed density matrix.")
        
        plot_wigner(reconstruction, xgrid, pgrid, fig=fig, ax=axs[1], label='reconstructed density matrix')
        plt.show()

    def plot_intermediate_hintons(self):
        """Plots Hinton diagrams of the density matrices in the progress_saves attribute."""
        if len(self.progress_saves[0].shape) == 2:
            reconstructions = self.progress_saves
        else:
            raise ValueError("Invalid shape of reconstructed density matrices.")

        subplot_number = _subplot_number(len(reconstructions))
        _, axs = plt.subplots(subplot_number[0], subplot_number[1], figsize=_subplot_figsize(len(reconstructions)), squeeze=False)
        axs = np.array(axs).flatten()
        for i, dm in enumerate(reconstructions):
            plot_hinton(dm, ax=axs[i], label=f"save {i}")
        plt.show()

    def plot_intermediate_Hintons(self):
        """Deprecated alias for plot_intermediate_hintons. Plots Hinton diagrams of the density matrices in the progress_saves attribute."""
        _deprecation_warning('plot_intermediate_Hintons', 'plot_intermediate_hintons')
        return self.plot_intermediate_hintons()

    def plot_intermediate_husimi_Qs(self, xgrid: np.ndarray, pgrid: np.ndarray):
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
            plot_husimi_Q(dm, xgrid, pgrid, fig=fig, ax=axs[i], label=f"save {i}")
        plt.show()

    def plot_intermediate_Husimi_Qs(self, xgrid: np.ndarray, pgrid: np.ndarray):
        """Deprecated alias for plot_intermediate_husimi_Qs. Plots the Husimi Q functions of the density matrices in the progress_saves attribute."""
        _deprecation_warning('plot_intermediate_Husimi_Qs', 'plot_intermediate_husimi_Qs')
        return self.plot_intermediate_husimi_Qs(xgrid, pgrid)
    
    def plot_intermediate_wigners(self, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Wigner functions of the density matrices in the progress_saves attribute.

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
            plot_wigner(dm, xgrid, pgrid, fig=fig, ax=axs[i], label=f"save {i}")
        plt.show()

class CustomQuantumStateTomography(QuantumStateTomography):
    """
    A class for designing custom quantum state tomography methods.
    
    Attributes
    ----------
    model : tf.keras.Model
        Model used for the reconstruction.
    training_step_fn : callable
        Function that defines the training step for the model. Arguments must include [model, measurement_data, measurement_operators]. Must return the generated density matrix and the loss.
    """
    def __init__(self, model, training_step_fn):
        super().__init__()
        self.model = model
        self.training_step_fn = training_step_fn

    def reconstruct(self, measurement_data, measurement_operators, epochs, optimizer, verbose_interval: int=None, num_progress_saves: int=None, true_dm: tf.Tensor=None, time_log_interval: int=None, **kwargs):
        """
        Reconstructs the density matrix using a custom method.

        Parameters
        ----------
        initial_dm : np.ndarray
            Initial density matrix.
        measurement_data : np.ndarray
            Frequency of each measurement outcome.
        measurement_operators : np.ndarray
            Projective operators corresponding to the measurement outcomes.
        epochs : int
            Number of training epochs.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer for the training step.
        verbose_interval : int
            Interval at which to print progress updates. Defaults to None.
        num_progress_saves : int
            Number of intermediate progress saves to make. Defaults to None.
        true_dm : tf.Tensor
            True density matrix used for calculating fidelities. Defaults to None.
        time_log_interval : int
            Interval at which to log the time taken after each epoch. Defaults to None.
        """
        # Input error handling
        if len(measurement_data[0]) != len(measurement_operators): raise ValueError("measurement_data[0] and measurement_operators must have the same length.")
        if not all([isinstance(data, np.ndarray) for data in measurement_data]): raise ValueError("All elements of measurement_data must be numpy arrays.")
        # if not all([isinstance(op, np.ndarray) for op in measurement_operators]): raise ValueError("All elements of measurement_operators must be numpy arrays.")
        if not isinstance(epochs, int): raise ValueError("epochs must be an integer.")
        if not isinstance(verbose_interval, int) and verbose_interval is not None: raise ValueError("verbose_interval must be an integer.")
        if not isinstance(num_progress_saves, int) and num_progress_saves is not None: raise ValueError("num_progress_saves must be an integer.")
        if not isinstance(time_log_interval, int) and time_log_interval is not None: raise ValueError("time_log_interval must be an integer.")
        
        self.optimizer = optimizer
        self.losses = []
        if num_progress_saves:
            progress_save_interval = epochs // num_progress_saves
            self.progress_saves = []
        else:
            self.progress_saves = None
        self.fidelities = [] if true_dm is not None else None
        if time_log_interval:
            start_time = time.time()
            self.times = []
        else:
            self.times = None

        for epoch in range(epochs):
            # Forward pass through generator
            with tf.GradientTape() as tape:
                generated_dm, epoch_loss = self.training_step_fn(self.model, measurement_data, measurement_operators, **kwargs)

            # Backpropagation
            gradients = tape.gradient(epoch_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            self.losses.append(epoch_loss.numpy())

            # Log fidelity
            if true_dm is not None:
                epoch_fidelity = fidelity(true_dm, generated_dm[0].numpy())
                self.fidelities.append(epoch_fidelity)

            # Save progress
            if num_progress_saves and epoch % progress_save_interval == 0:
                self.progress_saves.append(generated_dm[0].numpy())

            # Log progress
            if verbose_interval and epoch % verbose_interval == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss.numpy()}, Fidelity: {epoch_fidelity if true_dm is not None else None}")

            # Log time
            if time_log_interval and epoch % time_log_interval == 0:
                self.times.append(time.time() - start_time)

        self.reconstructed_dm = generated_dm[0].numpy()
        if verbose_interval: print('Reconstruction complete.')