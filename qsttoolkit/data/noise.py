import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from qutip import rand_dm


##### State preparation noise sources #####

def mixed_state_noise(density_matrix: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """
    Adds noise to a density matrix by mixing it with a random density matrix.

    Parameters
    ----------
    density_matrix : np.ndarray
        The density matrix to which noise will be added.
    noise_level : float, default=0.1
        The proportion of noise to add to the density matrix. Must be between 0 and 1.

    Returns
    -------
    np.ndarray
        The density matrix with noise added.
    """
    return (1 - noise_level) * density_matrix + noise_level * rand_dm(density_matrix.shape[0])

def gaussian_convolution(Q_function: np.ndarray, variance: float) -> np.ndarray:
    """
    Convolves a Q-function image with a Gaussian kernel.

    Parameters
    ----------
    Q_function : np.ndarray
        The Q-function image to be convolved.
    variance : float
        The variance of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        The Q-function image after convolution.
    """
    return gaussian_filter(Q_function, sigma=variance)


##### Experimental measurement noise sources #####

def affine_transformation(image: np.ndarray, theta: float, x: float, y: float) -> np.ndarray:
    """
    Applies an affine transformation to an image using TensorFlow's `apply_affine_transform` function.

    Parameters
    ----------
    image : np.ndarray
        The image to be transformed.
    theta : float
        The maximum rotation angle in degrees.
    x : float
        The maximum translation in the x direction.
    y : float
        The maximum translation in the y direction.
    
    Returns
    -------
    np.ndarray
        The transformed image.
    """
    theta = np.random.uniform(-theta, theta)
    x = np.random.uniform(-x, x)
    y = np.random.uniform(-y, y)
    return tf.keras.preprocessing.image.apply_affine_transform(np.stack([image] * 3, axis=-1), theta=theta, tx=x, ty=y, fill_mode='nearest')[:,:,0]

def additive_gaussian_noise(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Adds Gaussian noise to the image - sample from a Gaussian distribution with the given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        The image to which noise will be added.
    mean : float
        The mean of the Gaussian distribution.
    std : float
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        The image with Gaussian noise added.
    """
    noise = np.random.normal(mean, std, image.shape)
    noise[noise < 0] = 0
    image = image + noise
    return image

def salt_and_pepper_noise(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Adds salt-and-pepper noise to the image - set a proportion of pixels to 0.

    Parameters
    ----------
    image : np.ndarray
        The image to which noise will be added.
    prob : float
        The proportion of pixels to set to 0.
        
    Returns
    -------
    np.ndarray
        The image with salt-and-pepper noise added.
    """
    noise = np.random.rand(*image.shape)
    image[noise < prob] = 0
    return image


##### Combined noise #####

def apply_measurement_noise(image: np.ndarray, affine_theta: float, affine_x: float, affine_y: float, additive_Gaussian_stddev: float, salt_and_pepper_prob: float) -> np.ndarray:
    """
    Applies all types of measurement noise to the image, using the given parameters.
    
    Parameters
    ----------
    image : np.ndarray
        The image to which noise will be added.
    affine_theta : float
        The maximum rotation angle in degrees.
    affine_x : float
        The maximum translation in the x direction.
    affine_y : float
        The maximum translation in the y direction.
    additive_Gaussian_stddev : float
        The standard deviation of the Gaussian noise.
    salt_and_pepper_prob : float
        The proportion of pixels to set to 0.

    Returns
    -------
    np.ndarray
        The image with all types of noise added.
    """
    return salt_and_pepper_noise(
        additive_gaussian_noise(
            affine_transformation(image,
                                  affine_theta,
                                  affine_x,
                                  affine_y),
            np.mean(image),
            additive_Gaussian_stddev),
        salt_and_pepper_prob)