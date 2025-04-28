import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from qsttoolkit.tomography.QST import QuantumStateTomography, reconstruct_density_matrix
from qsttoolkit.tomography.dlqst.GAN_reconstructor.architecture import build_generator, build_discriminator
from qsttoolkit.tomography.dlqst.GAN_reconstructor.train import train
from qsttoolkit.utils import _deprecation_warning, _no_longer_required_warning


class GANQuantumStateTomography(QuantumStateTomography):
    """
    A class for training and evaluating a GAN for quantum state tomography.

    Attributes
    ----------
    data_dim : int
        Dimensions of the data vector.
    """
    def __init__(self, data_dim: int=None, latent_dim=None, dim=None):
        if dim: _no_longer_required_warning('dim')
        if data_dim is None:
            if latent_dim is not None:
                _deprecation_warning('latent_dim', 'data_dim')
                data_dim = latent_dim
            else:
                raise ValueError("data_dim must be specified.")

        super().__init__()
        self.generator = build_generator(data_vector_input_shape=[data_dim])
        self.discriminator = build_discriminator(data_vector_input_shape=[data_dim])

    def reconstruct(self, measurement_data: list, measurement_operators: list, epochs: int=100, gen_optimizer: tf.keras.optimizers.Optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), disc_optimizer: tf.keras.optimizers.Optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss_fn: tf.keras.losses.Loss=tf.keras.losses.BinaryCrossentropy(), verbose_interval: int=None, num_progress_saves: int=None, true_dm: tf.Tensor=None, time_log_interval: int=None):
        """
        Trains the GAN to reconstruct the density matrix from measurement data.

        Parameters
        ----------
        measurement_data : list of np.ndarray
            Frequency of each measurement outcome.
        measurement_operators : list of Qobj
            Projective operators corresponding to the measurement outcomes.
        epochs : int
            Number of epochs to train for. Defaults to 100.
        gen_optimizer : tf.keras.optimizers.Optimizer
            Generator optimizer. Defaults to Adam with learning rate 0.0002.
        disc_optimizer : tf.keras.optimizers.Optimizer
            Discriminator optimizer. Defaults to Adam with learning rate 0.0002.
        loss_fn : tf.keras.losses.Loss
            Loss function to use. Defaults to BinaryCrossentropy.
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
        if not isinstance(verbose_interval, int) and verbose_interval is not None: raise ValueError("verbose_interval must be an integer.")
        if not isinstance(num_progress_saves, int) and num_progress_saves is not None: raise ValueError("num_progress_saves must be an integer.")
        if not isinstance(time_log_interval, int) and time_log_interval is not None: raise ValueError("time_log_interval must be an integer.")

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        self.gen_losses, self.disc_losses, self.progress_saves, self.fidelities, self.times = train(self.generator,
                                                                                                    self.discriminator,
                                                                                                    measurement_data,
                                                                                                    measurement_operators,
                                                                                                    epochs=epochs,
                                                                                                    gen_optimizer=gen_optimizer,
                                                                                                    disc_optimizer=disc_optimizer,
                                                                                                    loss_fn=loss_fn,
                                                                                                    verbose_interval=verbose_interval,
                                                                                                    num_progress_saves=num_progress_saves,
                                                                                                    true_dm=true_dm,
                                                                                                    time_log_interval=time_log_interval)

        self.reconstructed_dm = reconstruct_density_matrix(self.generator(measurement_data))[0].numpy()
        if verbose_interval: print('Reconstruction complete.')

    def plot_losses(self):
        """Plots the generator and discriminator losses over epochs."""
        plt.figure(figsize=(5, 4))
        plt.plot(self.gen_losses, label='Generator loss')
        plt.plot(self.disc_losses, label='Discriminator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses over epochs')
        plt.show()

    def plot_loss_space(self):
        """Plots the loss functions against each other, coloured by the fidelities."""
        plt.figure(figsize=(10, 7))
        plt.plot(self.gen_losses, self.disc_losses, color='black', linewidth=0.5, alpha=0.7)
        scatter = plt.scatter(self.gen_losses, self.disc_losses, c=self.fidelities, cmap='Blues', s=20)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Fidelity', rotation=270, labelpad=15)
        plt.xlabel('Generator Loss')
        plt.ylabel('Discriminator Loss')
        all_values = self.gen_losses + self.disc_losses
        plt.xlim(min(all_values) - 0.005, max(all_values) + 0.005)
        plt.ylim(min(all_values) - 0.005, max(all_values) + 0.005)
        plt.title('Generator vs. Discriminator Losses Over Epochs')
        plt.grid()
        plt.show()