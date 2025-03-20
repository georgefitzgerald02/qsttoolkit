import numpy as np
from scipy.ndimage import gaussian_filter
from qutip import Qobj, rand_dm
import tensorflow as tf
import warnings


##### State preparation noise sources #####

def mixed_state_noise(density_matrix: Qobj, noise_level: float=0.1) -> np.ndarray:
    """
    Adds noise to a density matrix by mixing it with a random density matrix.

    Parameters
    ----------
    density_matrix : np.ndarray
        Density matrix to which noise will be added.
    noise_level : float
        Proportion of noise to add to the density matrix. Must be between 0 and 1. Defaults to 0.1.

    Returns
    -------
    np.ndarray
        Density matrix with noise added.
    """
    if type(density_matrix) == np.ndarray:
        density_matrix = Qobj(density_matrix)
    elif type(density_matrix) == tf.Tensor:
        density_matrix = Qobj(density_matrix.numpy())
    elif type(density_matrix) != Qobj:
        raise ValueError("unrecognised data type for density_matrix.")
    if noise_level < 0 or noise_level > 1:
        raise ValueError("noise_level must be between 0 and 1.")

    return (1 - noise_level) * density_matrix + noise_level * rand_dm(density_matrix.shape[0])

def gaussian_convolution(Q_function: np.ndarray, variance: float) -> np.ndarray:
    """
    Convolves a Q-function image with a Gaussian kernel.

    Parameters
    ----------
    Q_function : np.ndarray
        Q-function image to be convolved.
    variance : float
        Variance of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Q-function image after convolution.
    """
    return gaussian_filter(Q_function, sigma=variance)


##### Experimental measurement and data noise sources #####

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
    theta = np.random.uniform(-theta, theta)
    x = np.random.uniform(-x, x)
    y = np.random.uniform(-y, y)
    return tf.keras.preprocessing.image.apply_affine_transform(np.stack([image] * 3, axis=-1), theta=theta, tx=x, ty=y, fill_mode='nearest')[:,:,0]

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
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image[image < 0] = 0
    return image

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
    if not pepper_p and pepper_p != 0:
        if prob:
            pepper_p = prob
            warnings.warn("prob is deprecated and will be removed in a future version. Please use salt_p and pepper_p instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("pepper_p must be specified.")

    noise1 = np.random.rand(*image.shape)
    image[noise1 < salt_p] = 1
    noise2 = np.random.rand(*image.shape)
    image[noise2 < pepper_p] = 0
    return image


##### Combined noise #####

def apply_measurement_noise(image: np.ndarray, affine_theta: float, affine_x: float, affine_y: float, additive_Gaussian_stddev: float, pepper_p: float, salt_p: float=0.0, salt_and_pepper_prob=None) -> np.ndarray:
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
    if not pepper_p and pepper_p != 0:
        if salt_and_pepper_prob:
            pepper_p = salt_and_pepper_prob
            warnings.warn("prob is deprecated and will be removed in a future version. Please use salt_p and pepper_p instead.", DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("pepper_p must be specified.")

    return salt_and_pepper_noise(
        additive_gaussian_noise(
            affine_transformation(image,
                                  affine_theta,
                                  affine_x,
                                  affine_y),
            np.mean(image),
            additive_Gaussian_stddev),
        pepper_p, salt_p)