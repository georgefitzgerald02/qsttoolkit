import numpy as np
import cmath
import random
import tensorflow as tf


##### Basic #####

def _random_complex(magnitude_range: list[float, float]) -> complex:
    """Generates a random complex number with a given magnitude range."""
    # Error checking
    _range_error(magnitude_range)
    
    magnitude = random.uniform(magnitude_range[0], magnitude_range[1])      # Is uniform best?? Options?
    angle = random.uniform(0, 2 * cmath.pi)
    return cmath.rect(magnitude, angle)

def _get_var_name(var: any) -> str:
    """Returns the name of a variable."""
    for name, value in globals().items():
        if value is var:
            return name
        
def _subplot_number(n_subplots: int, max_h_subplots: int=5) -> tuple:
    """Returns a tuple for the number of rows and columns of a plot with n_subplots subplots."""
    h_subplots = n_subplots if n_subplots < max_h_subplots else max_h_subplots
    v_subplots = int(np.ceil(n_subplots / h_subplots))
    return (v_subplots, h_subplots)
        
def _subplot_figsize(n_subplots: int, subplot_width: int=5, max_h_subplots: int=5) -> tuple:
    """Returns a tuple for the figsize of a plot with n_subplots subplot_width x subplot_width size subplots."""
    # width - n_subplots if less than max_h_subplots, else max_h_subplots
    width = n_subplots * subplot_width if n_subplots < max_h_subplots else max_h_subplots * subplot_width
    # height - round up to nearest multiple of max_h_subplots
    height = int(np.ceil(n_subplots / max_h_subplots)) * subplot_width
    return (width, height)
        

##### Regularisation #####

def _L1_regularisation(weights: tf.Tensor, L1_reg: float) -> tf.Tensor:
    """Calculates the L1 regularisation term for a tf.Tensor."""
    return L1_reg * tf.reduce_sum(tf.abs(weights))

def _threshold_regularisation(weights: tf.Tensor, threshold: float, thresh_reg: float) -> tf.Tensor:
    """Calculates the threshold regularisation term for a tf.Tensor."""
    above_threshold = tf.math.greater(tf.math.abs(weights), threshold)
    num_above_threshold = tf.math.reduce_sum(tf.cast(above_threshold, tf.float64))
    return thresh_reg * num_above_threshold


##### Error handling #####

def _range_error(magnitude_range: list[float, float], integers: bool=False, positive: bool=True):
    """Basic error checking for the range of magnitudes."""
    if len(magnitude_range) != 2:
        raise ValueError("magnitude_range must be a list of two floats")
    if magnitude_range[0] > magnitude_range[1]:
        raise ValueError("min magnitude must be less than max magnitude")
    if positive:
        if magnitude_range[0] < 0:
            raise ValueError("min magnitude must be greater than or equal to 0")
    if integers:
        if not float(magnitude_range[0]).is_integer() or not float(magnitude_range[1]).is_integer():
            raise ValueError("min and max magnitudes must be integers")