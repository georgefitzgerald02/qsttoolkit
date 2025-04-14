import tensorflow as tf
from tensorflow.keras import layers, Model


##### Define custom layers #####

class DensityMatrix(layers.Layer):                  # No longer in use - generator now produces a Cholesky decomposition rather than a full density matrix
    """Custom layer to convert a 2D complex-valued tensor into a density matrix. The input tensor is reshaped into a lower triangular matrix, and the diagonal is set to be real."""
    def __init__(self, **kwargs):
        super(DensityMatrix, self).__init__(**kwargs)

    def call(self, input):
        """Define the forward pass logic."""
        input_matrix = tf.complex(input[..., 0], input[..., 1])
        self.dim = input_matrix.shape[-1]
        lower_triangular = tf.linalg.band_part(input_matrix, -1, 0)
        real_diag = tf.cast(tf.math.real(tf.linalg.diag_part(lower_triangular)), dtype=tf.complex64)
        T = tf.linalg.set_diag(lower_triangular, real_diag)
        rho = tf.matmul(tf.linalg.adjoint(T), T)
        rho /= tf.linalg.trace(rho)
        # Split into real and imaginary parts
        # rho_stacked = tf.stack([tf.math.real(rho), tf.math.imag(rho)], axis=-1)
        return tf.cast(rho, dtype=tf.complex128)[0]

    def compute_output_shape(self):
        """Compute the output shape of the layer."""
        return (self.dim, self.dim)

    def get_config(self):
        """Save the layer configuration for serialisation."""
        config = super(DensityMatrix, self).get_config()
        config.update({"dim": self.dim})
        return config
    
class CholeskyLowerTriangular(layers.Layer):
    """Custom layer to convert a 2D complex-valued tensor into a lower triangular Cholesky decomposition. input tensor is reshaped into a lower triangular matrix, and the diagonal is set to be real."""
    def __init__(self, **kwargs):
        super(CholeskyLowerTriangular, self).__init__(**kwargs)

    def call(self, input):
        """Define the forward pass logic."""
        input_matrix = tf.complex(input[..., 0], input[..., 1])
        lower_triangular = tf.cast(tf.linalg.band_part(input_matrix, -1, 0), dtype=tf.complex128)
        real_diag = tf.cast(tf.math.real(tf.linalg.diag_part(lower_triangular)), dtype=tf.complex128)
        T = tf.linalg.set_diag(lower_triangular, real_diag)
        return T

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape  # shape remains the same

    def get_config(self):
        """Save the layer configuration for serialisation."""
        config = super(CholeskyLowerTriangular, self).get_config()
        return config


##### Model architecture functions #####

def build_generator(data_vector_input_shape: tuple) -> tf.keras.Model:
    """
    Builds the generator to reconstruct a density matrix Cholesky decomposition from measurement data vectors.

    Parameters
    ----------
    data_vector_input_shape : tuple
        Shape of the input measurement data vector.
    noise_parameters_input_shape : list
        Shape of the input noise parameters. Defaults to []. Currently not implemented.

    Returns
    -------
    tf.keras.Model
        Generator model.
    """
    data_vector_input = layers.Input(shape=data_vector_input_shape, name='data_vector_input')

    x = layers.Dense(512)(data_vector_input)
    x = layers.LeakyReLU()(x)

    x = layers.Reshape((16, 16, 2))(x)

    x = layers.Conv2DTranspose(64, 4, 2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, 4, 1, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(32, 4, 1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(2, 4, 1, padding='same')(x)
    lower_triangular = CholeskyLowerTriangular()(x)

    return Model(inputs=data_vector_input, outputs=lower_triangular)

def build_discriminator(data_vector_input_shape: tuple) -> tf.keras.Model:
    """
    Builds the discriminator to classify the reconstructed density matrices.

    Parameters
    ----------
    data_vector_input_shape : tuple
        Shape of the input measurement data vector.

    Returns
    -------
    tf.keras.Model
        Discriminator model.
    """
    data_vector_input = layers.Input(shape=data_vector_input_shape, name='data_vector_input')

    x = layers.Dense(128)(data_vector_input)
    x = layers.LeakyReLU()(x)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dense(32)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=data_vector_input, outputs=x)