from qutip import Qobj
import tensorflow as tf

from qsttoolkit.quantum import expectation
from qsttoolkit.utils import _L1_regularisation, _threshold_regularisation, _no_longer_required_warning


def log_likelihood(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: list, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01, dim=None) -> tf.Tensor:
    """
    Computes the negative log-likelihood of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
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
    if dim is not None: _no_longer_required_warning('dim')

    if type(rho) == Qobj: rho = rho.full()

    # Compute probabilities: p_k = Tr(E_k * rho) for all projectors
    probabilities = expectation(rho, measurement_operators)
    probabilities /= tf.reduce_sum(probabilities)  # Normalize to ensure sum = 1

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute log-likelihood
    log_likelihood = tf.reduce_sum(frequency_data * tf.math.log(probabilities))

    return -log_likelihood + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)

def mean_absolute_error(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: list, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01) -> tf.Tensor:
    """
    Computes the mean absolute error of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
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
        Mean absolute error of the data given the density matrix.
    """
    if type(rho) == Qobj: rho = rho.full()

    # Compute probabilities: p_k = Tr(P_k * rho) for all projectors
    probabilities = expectation(rho, measurement_operators)
    probabilities /= tf.reduce_sum(probabilities)  # Normalize to ensure sum = 1

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute mean absolute error
    mae = tf.reduce_mean(tf.abs(frequency_data - probabilities))

    return mae + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)

def mean_squared_error(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: list, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01) -> tf.Tensor:
    """
    Computes the mean squared error of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
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
        Mean squared error of the data given the density matrix.
    """
    if type(rho) == Qobj: rho = rho.full()

    # Compute probabilities: p_k = Tr(P_k * rho) for all projectors
    probabilities = expectation(rho, measurement_operators)
    probabilities /= tf.reduce_sum(probabilities)  # Normalize to ensure sum = 1

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute mean squared error
    mse = tf.reduce_mean(tf.square(frequency_data - probabilities))

    return mse + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)

def kl_divergence(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: list, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01) -> tf.Tensor:
    """
    Computes the Kullback-Leibler divergence of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
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
        Kullback-Leibler divergence of the data given the density matrix.
    """
    if type(rho) == Qobj: rho = rho.full()

    # Compute probabilities: p_k = Tr(P_k * rho) for all projectors
    probabilities = expectation(rho, measurement_operators)
    probabilities /= tf.reduce_sum(probabilities)  # Normalize to ensure sum = 1

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute KL divergence
    kl_div = tf.reduce_sum(frequency_data * (tf.math.log(frequency_data / probabilities)))

    return kl_div + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)

def squared_hellinger_distance(rho: tf.Tensor, frequency_data: tf.Tensor, measurement_operators: list, L1_reg: float=0, thresh_reg: float=0, thresh_reg_threshold: float=0.01) -> tf.Tensor:
    """
    Computes the squared Hellinger distance of the data given the density matrix.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix.
    frequency_data : tf.Tensor
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
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
        Squared Hellinger distance of the data given the density matrix.
    """
    if type(rho) == Qobj: rho = rho.full()

    # Compute probabilities: p_k = Tr(P_k * rho) for all projectors
    probabilities = expectation(rho, measurement_operators)
    probabilities /= tf.reduce_sum(probabilities)  # Normalize to ensure sum = 1

    # Ensure probabilities are numerically stable (avoid log(0))
    probabilities = tf.clip_by_value(probabilities, 1.0e-10, 1.0)

    # Compute squared Hellinger distance
    hellinger_distance = tf.reduce_sum(tf.square(tf.sqrt(frequency_data) - tf.sqrt(probabilities)))

    return hellinger_distance + _L1_regularisation(rho, L1_reg) + _threshold_regularisation(rho, thresh_reg_threshold, thresh_reg)