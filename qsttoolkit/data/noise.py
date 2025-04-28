import numpy as np
from scipy.ndimage import gaussian_filter
from qutip import Qobj, rand_dm, destroy, mesolve, thermal_dm
import tensorflow as tf
import warnings

from qsttoolkit.utils import _estimate_min_nsteps, _deprecation_warning


# CONTENTS:
# 1. State preparation noise
# 2. Measurement noise
#   a. Shot noise
#   c. Heterodyne detection
# 3. Data noise
#   a. Photodetector noise
# 4. Combined noise

##### State preparation noise #####

def mixed_state_noise(rho: Qobj, noise_level: float=0.1, density_matrix=None) -> np.ndarray:
    """
    Adds noise to a density matrix by mixing it with a random density matrix.

    Parameters
    ----------
    rho : Qobj
        Density matrix to which noise will be added.
    noise_level : float
        Proportion of noise to add to the density matrix. Must be between 0 and 1. Defaults to 0.1.

    Returns
    -------
    np.ndarray
        Density matrix with noise added.
    """
    if density_matrix is not None:
        _deprecation_warning('density_matrix', 'rho')
        rho = density_matrix
    if type(rho) == np.ndarray:
        rho = Qobj(rho)
    elif type(rho) == Qobj:
        rho = Qobj(rho.full())    # Convert to full matrix if Qobj
    else:
        raise ValueError("unrecognized data type for rho.")
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise_level must be a float between 0 and 1.")

    return (1 - noise_level) * rho + noise_level * rand_dm(rho.shape[0])

def photon_loss_noise(rho: Qobj, resonator_frequency: float, loss_rate: float, tau: float, nsteps: int=None, nstep_increase_factor: float=2.0, verbose: bool=False) -> np.ndarray:
    """
    Simulates photon loss in the resonator by evolving the density matrix using the Lindbladian master equation.

    Parameters
    ----------
    rho : Qobj
        Density matrix to which noise will be added.
    resonator_frequency : float
        Frequency of the resonator.
    loss_rate : float
        Rate of photon loss.
    tau : float
        Time duration of the evolution.
    nsteps : int, optional
        Number of steps for the evolution. If None, it will be estimated based on the resonator frequency and loss rate.
    nstep_increase_factor : float, optional
        Factor by which to increase the number of steps if the evolution fails. Defaults to 2.0.
    verbose : bool, optional
        If True, prints additional information during the evolution. Defaults to False.

    Returns
    -------
    np.ndarray
        Density matrix after photon loss noise has been applied.
    """
    if type(rho) == np.ndarray:
        rho = Qobj(rho)
    elif type(rho) == Qobj:
        rho = Qobj(rho.full())    # Convert to full matrix if Qobj
    else:
        raise ValueError("unrecognized data type for rho.")
    if not isinstance(resonator_frequency, (int, float)) or resonator_frequency < 0: raise ValueError("resonator_frequency must be a positive float.")
    if not isinstance(loss_rate, (int, float)) or loss_rate < 0: raise ValueError("loss_rate must be a positive float.")
    if not isinstance(tau, (int, float)) or tau < 0: raise ValueError("tau must be a positive float.")
    if not isinstance(nsteps, (int, float)) and nsteps is not None: raise ValueError("nsteps must be an integer.")
    if not isinstance(nstep_increase_factor, (int, float)): raise ValueError("nstep_increase_factor must be a float.")
    if nsteps is not None and nsteps < 0: raise ValueError("nsteps must be a positive integer.")
    if nstep_increase_factor < 1: raise ValueError("nstep_increase_factor must be a positive float.")

    dim = rho.shape[0]
    a = destroy(dim)  # Annihilation operator
    H = resonator_frequency * a.dag() * a  # Free resonator Hamiltonian
    collapse_ops = [np.sqrt(loss_rate) * a]  # Collapse operator for photon loss

    if nsteps is None: nsteps = _estimate_min_nsteps(resonator_frequency, tau, loss_rate)
    while True:
        try:
            if verbose: print(f"Simulating photon loss noise in {nsteps} steps.")
            evolved_state = mesolve(H, rho, [0, tau], collapse_ops, e_ops=[], options={'nsteps': nsteps}) # Simulate the evolution of the density matrix using the Lindbladian master equation
            if verbose: print(f"Photon loss noise simulated in {nsteps} steps.")
            break
        except Exception:
            nsteps *= nstep_increase_factor
            if verbose: print(f"Error during evolution, increasing nsteps to {nsteps}.")
    return evolved_state.states[-1]  # Return the final state after evolution


##### Measurement noise #####

def additive_gaussian_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Adds Gaussian noise to the image by sampling from a Gaussian distribution with the given mean and standard deviation. This type of noise arises from finite measurements and discrete binning of continuous data.

    Parameters
    ----------
    image : np.ndarray
        Image to which noise will be added.
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        Image with Gaussian noise added.
    """
    if type(image) != np.ndarray: raise ValueError("unrecognized data type for Q_function, expected np.ndarray.")
    if std < 0: raise ValueError("std must be a positive float.")

    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image[image < 0] = 0
    return image

### Heterodyne detection ###

def mode_mismatch_noise(alpha_grid: np.ndarray, mu: float) -> np.ndarray:
    """
    Simulates mode mismatch noise by scaling the amplitude of the complex amplitude alpha.

    Parameters
    ----------
    alpha_grid : np.ndarray
        Phase space grid of complex amplitudes.
    mu : float
        Scaling factor for the amplitude.
    
    Returns
    -------
    np.ndarray
        Phase space grid with mode mismatch noise applied.
    """
    if type(alpha_grid) != np.ndarray: raise ValueError("unrecognized data type for alpha_grid, expected np.ndarray.")
    if not isinstance(mu, (int, float)) or mu < 0: raise ValueError("mu must be a positive float.")

    return np.sqrt(mu) * alpha_grid

def phase_error_noise(alpha_grid: np.ndarray, sigma_rad: float) -> np.ndarray:
    """
    Simulates phase error noise by adding a Gaussian noise to the phase of the complex amplitude alpha.

    Parameters
    ----------
    alpha_grid : np.ndarray
        Phase space grid of complex amplitudes.
    sigma_rad : float
        Standard deviation of the Gaussian noise in radians.

    Returns
    -------
    np.ndarray
        Phase space grid with phase error noise added.
    """
    if type(alpha_grid) != np.ndarray: raise ValueError("unrecognized data type for alpha_grid, expected np.ndarray.")
    if not isinstance(sigma_rad, (int, float)) or sigma_rad < 0: raise ValueError("sigma_rad must be a positive float.")

    phase_noise = np.random.normal(0, sigma_rad, size=alpha_grid.shape)
    return alpha_grid * np.exp(1j * phase_noise)

def displacement_error_noise(alpha_grid: np.ndarray, amp_std: float) -> np.ndarray:
    """
    Simulates displacement error noise by adding a Gaussian noise to the amplitude of the complex amplitude alpha.

    Parameters
    ----------
    alpha_grid : np.ndarray
        Phase space grid of complex amplitudes.
    amp_std : float
        Standard deviation of the Gaussian noise in amplitude.

    Returns
    -------
    np.ndarray
        Phase space grid with displacement error noise added.
    """
    if type(alpha_grid) != np.ndarray: raise ValueError("unrecognized data type for alpha_grid, expected np.ndarray.")
    if not isinstance(amp_std, (int, float)) or amp_std < 0: raise ValueError("amp_std must be a positive float.")

    amp_noise = np.random.normal(0, amp_std, size=alpha_grid.shape) + 1j * np.random.normal(0, amp_std, size=alpha_grid.shape)
    return alpha_grid + amp_noise

def affine_transformation(image: np.ndarray, theta: float, x: float, y: float) -> np.ndarray:
    """
    Applies a random affine transformation to an image using TensorFlow's `apply_affine_transform` function.

    Parameters
    ----------
    image : np.ndarray
        Image to be transformed.
    theta : float
        Maximum rotation angle in degrees.
    x : float
        Maximum translation in the x direction.
    y : float
        Maximum translation in the y direction.
    
    Returns
    -------
    np.ndarray
        Transformed image.
    """
    warnings.simplefilter("ignore", DeprecationWarning)  # Suppress warning from faulty SciPy deprecation error
    if type(image) != np.ndarray: raise ValueError("unrecognized data type for Q_function, expected np.ndarray.")

    theta = np.random.uniform(-theta, theta)
    x = np.random.uniform(-x, x)
    y = np.random.uniform(-y, y)
    image = tf.keras.preprocessing.image.apply_affine_transform(np.stack([image] * 3, axis=-1), theta=theta, tx=x, ty=y, fill_mode='nearest')[:,:,0]
    return image


##### Data noise #####
### Photodetector noise ###

def amplification_noise(Q_function: np.ndarray, ntherm: float=None, variance=None) -> np.ndarray:
    """
    Simulates additional bosonic modes from photon detection using linear amplifiers. This is done by convolving the Q-function measurement image with a Gaussian kernel.

    Parameters
    ----------
    Q_function : np.ndarray
        Q-function image to be convolved.
    ntherm : float
        Variance of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Q-function image after convolution.
    """
    if variance is not None:
        _deprecation_warning('variance', 'ntherm')
        ntherm = variance
    if type(Q_function) != np.ndarray: raise ValueError("unrecognized data type for Q_function, expected np.ndarray.")
    if ntherm < 0: raise ValueError("ntherm must be a positive float.")

    image = gaussian_filter(Q_function, sigma=ntherm)
    return image

def gaussian_convolution(Q_function: np.ndarray, variance: float) -> np.ndarray:
    """Deprecated alias for amplification_noise."""
    _deprecation_warning('gaussian_convolution', 'amplification_noise')
    return amplification_noise(Q_function, variance)

def dark_count_noise(measurement_operators: list, dark_count_p: float, n_thermal=0.1) -> list:
    """
    Simulates dark counts by adding a thermal state to a list of measurement operators.

    Parameters
    ----------
    measurement_operators : list of Qobj of Qobj
        List of measurement operators to which noise will be added.
    dark_count_p : float
        Probability of dark counts.
    n_thermal : float
        Average number of thermal photons. Defaults to 0.1.

    Returns
    -------
    list
        List of measurement operators that simulate dark counts.
    """
    # if type(measurement_operators) != list: raise ValueError("unrecognized data type for measurement_operators, expected list.")
    if not isinstance(dark_count_p, (int, float)) or dark_count_p < 0 or dark_count_p > 1: raise ValueError("dark_count_p must be a float between 0 and 1.")
    if not isinstance(n_thermal, (int, float)) or n_thermal < 0: raise ValueError("n_thermal must be a positive float.")

    dim = measurement_operators[0].shape[0]
    thermal_noise = thermal_dm(dim, n_thermal)
    return [(1 - dark_count_p) * Ek + dark_count_p * thermal_noise for Ek in measurement_operators]

def salt_and_pepper_noise(image: np.ndarray, pepper_p: float, salt_p: float=0.0, prob=None) -> np.ndarray:
    """
    Adds salt-and-pepper noise to the image - set a proportion of pixels to 0.

    Parameters
    ----------
    image : np.ndarray
        Image to which noise will be added.
    pepper_p : float
        Proportion of pixels to set to 0.
    salt_p : float
        Proportion of pixels to set to 1. Defaults to 0.0.
        
    Returns
    -------
    np.ndarray
        Image with salt-and-pepper noise added.
    """
    if type(image) != np.ndarray: raise ValueError("unrecognized data type for Q_function, expected np.ndarray.")
    if not pepper_p and pepper_p != 0:
        if prob:
            pepper_p = prob
            warnings.warn("'prob' is deprecated and will be removed in a future version. Please use 'salt_p' and 'pepper_p' instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("pepper_p must be specified.")
    if salt_p < 0 or salt_p > 1: raise ValueError("salt_p must be a float between 0 and 1.")
    if pepper_p < 0 or pepper_p > 1: raise ValueError("pepper_p must be a float between 0 and 1.")

    noise1 = np.random.rand(*image.shape)
    image[noise1 < salt_p] = 1
    noise2 = np.random.rand(*image.shape)
    image[noise2 < pepper_p] = 0
    return image


##### Combined noise #####

def apply_measurement_noise(image: np.ndarray, affine_theta: float, affine_x: float, affine_y: float, additive_Gaussian_stddev: float, pepper_p: float, salt_p: float=0.0, amplification_ntherm: float=0.0, salt_and_pepper_prob=None) -> np.ndarray:
    """
    Applies all types of measurement noise to the image, using the given parameters.
    
    Parameters
    ----------
    image : np.ndarray
        Image to which noise will be added.
    affine_theta : float
        Maximum rotation angle in degrees.
    affine_x : float
        Maximum translation in the x direction.
    affine_y : float
        Maximum translation in the y direction.
    amplification_ntherm : float
        Variance of the Gaussian kernel used for amplification noise.
    additive_Gaussian_stddev : float
        Standard deviation of the Gaussian distribution from which additive noise is sampled.
    salt_p : float
        Proportion of pixels to set to 1. Defaults to 0.
    pepper_p : float
        Proportion of pixels to set to 0.

    Returns
    -------
    np.ndarray
        Image with all types of noise added.
    """
    if type(image) != np.ndarray: raise ValueError("unrecognized data type for Q_function, expected np.ndarray.")
    if not pepper_p and pepper_p != 0:
        if salt_and_pepper_prob:
            pepper_p = salt_and_pepper_prob
            warnings.warn("'prob' is deprecated and will be removed in a future version. Please use 'salt_p' and 'pepper_p' instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("pepper_p must be specified.")

    image = affine_transformation(image, affine_theta, affine_x, affine_y)
    image = amplification_noise(image, amplification_ntherm)
    image = additive_gaussian_noise(image, 0.0, additive_Gaussian_stddev)
    image = salt_and_pepper_noise(image, pepper_p, salt_p)
    return image