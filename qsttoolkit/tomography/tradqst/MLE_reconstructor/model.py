import numpy as np
from qutip import Qobj
import tensorflow as tf
import warnings

from qsttoolkit.tomography.QST import QuantumStateTomography, parametrize_density_matrix, reconstruct_density_matrix
from qsttoolkit.tomography.tradqst.MLE_reconstructor.train import train


class MLEQuantumStateTomography(QuantumStateTomography):
    """A class for performing maximum likelihood estimation quantum state tomography."""
    def __init__(self, dim=None):
        if dim: warnings.warn("dim is no longer required for this class and will be removed in a future version.", DeprecationWarning, stacklevel=2)

        super().__init__()
        self.params = tf.Variable([], dtype=tf.complex128)

    def reconstruct(self, measurement_data: list, measurement_operators: list, initial_dm: np.ndarray, epochs: int=500, optimizer: tf.keras.optimizers.Optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), L1_reg: float=0.0, verbose_interval: int=None, num_progress_saves: int=None, true_dm: tf.Tensor=None, time_log_interval: int=None, method=None, verbose=None):
        """
        Fits the density matrix to the measurement data using maximum likelihood estimation.

        Parameters
        ----------
        measurement_data : list of np.ndarray
            Frequency of each measurement outcome.
        measurement_operators : list of np.ndarray
            Projective operators corresponding to the measurement outcomes.
        initial_dm : np.ndarray
            Initial density matrix guess.
        epochs : int
            Number of optimization epochs. Defaults to 500.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer to use for the reconstruction. Defaults to Adam with learning rate 0.01.
        L1_reg : float
            L1 regularisation parameter. Defaults to 0.
        verbose_interval : int
            Interval at which to print progress updates. Defaults to None.
        num_progress_saves : int
            Number of intermediate progress saves to make. Defaults to None.
        true_dm : tf.Tensor
            True density matrix used for calculating fidelities. Defaults to None.
        time_log_interval : int
            Interval at which to log the time taken after each epoch. Defaults to None.
        """
        if method: warnings.warn("method is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2)
        if verbose:
            warnings.warn("verbose is deprecated and will be removed in a future version. Please use verbose_interval instead.", DeprecationWarning, stacklevel=2)

        # Input error handling
        if len(measurement_data[0]) != len(measurement_operators):
            raise ValueError("measurement_data[0] and measurement_operators must have the same length.")
        if not all([isinstance(data, np.ndarray) for data in measurement_data]):
            raise ValueError("All elements of measurement_data must be numpy arrays.")
        if not all([isinstance(op, np.ndarray) for op in measurement_operators]):
            raise ValueError("All elements of measurement_operators must be numpy arrays.")
        if not isinstance(verbose_interval, int) and verbose_interval is not None:
            raise ValueError("verbose_interval must be an integer.")
        if not isinstance(num_progress_saves, int) and num_progress_saves is not None:
            raise ValueError("num_progress_saves must be an integer.")
        if not isinstance(time_log_interval, int) and time_log_interval is not None:
            raise ValueError("time_log_interval must be an integer.")
        
        # Ensure initial density matrix is array-like
        if type(initial_dm) == Qobj:
            initial_dm = initial_dm.full()
        elif type(initial_dm) == tf.Tensor:
            initial_dm = initial_dm.numpy()

        self.initial_dm = tf.cast(initial_dm, dtype=tf.complex128)
        # Convert initial density matrix into trainable Cholesky decomposition
        self.params = tf.Variable(parametrize_density_matrix(self.initial_dm), dtype=tf.complex128)

        self.optimizer = optimizer

        self.losses, self.progress_saves, self.fidelities, self.times = train(self.params,
                                                                              measurement_data,
                                                                              measurement_operators,
                                                                              epochs=epochs,
                                                                              optimizer=self.optimizer,
                                                                              L1_reg=L1_reg,
                                                                              verbose_interval=verbose_interval,
                                                                              num_progress_saves=num_progress_saves,
                                                                              true_dm=true_dm,
                                                                              time_log_interval=time_log_interval)

        # Convert final parameters back to density matrix
        self.reconstructed_dm = reconstruct_density_matrix(self.params).numpy()
        if verbose_interval: print("Reconstruction complete.")

    def plot_cost_values(self):
        """Deprecated alias for plot_losses."""
        warnings.warn("plot_cost_values is deprecated and will be removed in a future version. Please use plot_losses instead.", DeprecationWarning, stacklevel=2)
        self.plot_losses()