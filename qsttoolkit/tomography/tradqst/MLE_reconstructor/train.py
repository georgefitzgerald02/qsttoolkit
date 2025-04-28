import tensorflow as tf
import time

from qsttoolkit.tomography.QST import reconstruct_density_matrix, log_likelihood
from qsttoolkit.quantum import fidelity


def train_step(params: tf.Variable, measurement_data: list, measurement_operators: list, optimizer: tf.keras.optimizers.Optimizer, L1_reg: float=0.0):
    """
    Performs one optimization step using gradient descent.
    
    Parameters
    ----------
    params : tf.Variable
        Trainable parametrization of the density matrix.
    measurement_data : list of np.ndarray
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
        Projective operators corresponding to the measurement outcomes.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to use for the reconstruction.
    L1_reg : float
        L1 regularisation parameter. Defaults to 0.
    """
    with tf.GradientTape() as tape:
        rho = reconstruct_density_matrix(params)
        loss = log_likelihood(rho, measurement_data, measurement_operators, L1_reg=L1_reg)

    gradients = tape.gradient(loss, [params])
    optimizer.apply_gradients(zip(gradients, [params]))
    
    return loss

def train(params: tf.Variable, measurement_data: list, measurement_operators: list, epochs: int=100, optimizer: tf.keras.optimizers.Optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), L1_reg: float=0.0, verbose_interval: int=None, num_progress_saves: int=None, true_dm: tf.Tensor=None, time_log_interval: int=None) -> tuple:
    """
    Fits the density matrix to the measurement data using maximum likelihood estimation.
    
    Parameters
    ----------
    params : tf.Variable
        Trainable parametrization of the density matrix.
    measurement_data : list of np.ndarray
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
        Projective operators corresponding to the measurement outcomes.
    epochs : int
        Number of optimization epochs. Defaults to 100.
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

    Returns
    -------
    list of float
        Losses after each epoch.
    list of tf.Tensor
        Intermediate progress saves.
    list of float
        Fidelities of the generator density matrices with respect to the true density matrix.
    list of float
        Time taken after each epoch.
    """
    losses = []
    if num_progress_saves:
        progress_save_interval = epochs // num_progress_saves
        progress_saves = []
    else:
        progress_saves = None
    fidelities = [] if true_dm is not None else None
    if time_log_interval:
        start_time = time.time()
        times = []
    else:
        times = None

    for epoch in range(epochs):
        loss_value = train_step(params, measurement_data, measurement_operators, optimizer, L1_reg)

        # Save losses
        losses.append(loss_value.numpy())

        # Save fidelities
        if true_dm is not None:
            fidelities.append(fidelity(true_dm, reconstruct_density_matrix(params)))

        # Save progress
        if num_progress_saves and epoch % progress_save_interval == 0:
            progress_saves.append(reconstruct_density_matrix(params).numpy())

        # Log progress
        if verbose_interval and epoch % verbose_interval == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss_value.numpy()}, Fidelity: {fidelities[-1] if true_dm is not None else None}")
        
        # Log time
        if time_log_interval and epoch % time_log_interval == 0:
            times.append(time.time() - start_time)
    
    return losses, progress_saves, fidelities, times