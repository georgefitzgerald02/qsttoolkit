import numpy as np
import matplotlib.pyplot as plt

from qsttoolkit.tomography.dlqst.GAN_reconstructor.architecture import build_generator, build_discriminator
from qsttoolkit.tomography.dlqst.GAN_reconstructor.train import train
from qsttoolkit.plots import plot_hinton, plot_Husimi_Q


class GANQuantumStateTomography:
    """
    A class for training and evaluating a GAN for quantum state tomography.

    Attributes
    ----------
    dim : int
        The Hilbert space dimensionality.
    generator : tf.keras.Model
        The generator model.
    discriminator : tf.keras.Model
        The discriminator model.
    gen_losses : list
        The generator losses over epochs.
    disc_losses : list
        The discriminator losses over epochs.
    progress_saves : list
        The progress saves.
    fidelities : list
        The fidelities over epochs.
    reconstructed_dm : np.ndarray
        The reconstructed density matrix.
    """
    def __init__(self, dim: int, latent_dim: int):
        self.dim = dim
        self.generator = build_generator(data_vector_input_shape=(latent_dim**2,), dim=self.dim)
        self.discriminator = build_discriminator(data_vector_input_shape=(latent_dim**2,))

    def reconstruct(self, measurement_data: np.ndarray, measurement_operators: np.ndarray, epochs=1000, verbose_interval=None, num_progress_saves=None, true_dm=None):
        """
        Trains the GAN to reconstruct the density matrix from measurement data.

        Parameters
        ----------
        measurement_data : np.ndarray
            The measurement data.
        measurement_operators : np.ndarray
            The projective measurement operators.
        epochs : int, optional
            The number of epochs to train for. Defaults to 1000.
        verbose_interval : int, optional
            The interval at which to print the losses. Defaults to None.
        num_progress_saves : int, optional
            The number of progress saves to make. Defaults to None.
        true_dm : np.ndarray, optional
            The true density matrix. Defaults to None.
        """
        self.gen_losses, self.disc_losses, self.progress_saves, self.fidelities = train(self.generator,
                                                                                        self.discriminator,
                                                                                        measurement_data,
                                                                                        measurement_operators,
                                                                                        epochs=epochs,
                                                                                        verbose_interval=verbose_interval,
                                                                                        num_progress_saves=num_progress_saves,
                                                                                        true_dm=true_dm)

        self.reconstructed_dm = self.generator(measurement_data).numpy()
        print('Density matrix reconstruction complete')

    def plot_losses(self):
        """
        Plots the generator and discriminator losses over epochs.
        """
        plt.figure(figsize=(5, 4))
        plt.plot(self.gen_losses, label='Generator loss')
        plt.plot(self.disc_losses, label='Discriminator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Losses over epochs')
        plt.show()

    def plot_fidelities(self):
        """
        Plots the fidelities over epochs.
        """
        plt.figure(figsize=(5, 4))
        plt.plot(self.fidelities)
        plt.ylim(0,1)
        plt.xlabel('Epoch')
        plt.ylabel('Fidelity')
        plt.title('Fidelity over epochs')
        plt.show()

    def plot_comparison_hintons(self, true_dm: np.ndarray):
        """
        Plots the Hinton diagrams of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            The true density matrix.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_hinton(true_dm, ax=axs[0], label='true density matrix')
        plot_hinton(self.reconstructed_dm, ax=axs[1], label='optimised density matrix')
        plt.show()

    def plot_comparison_Husimi_Qs(self, true_dm: np.ndarray, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Husimi-Q functions of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            The true density matrix.
        xgrid : np.ndarray
            The phase space x grid.
        pgrid : np.ndarray
            The phase space p grid.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_Husimi_Q(true_dm, xgrid, pgrid, fig=fig, ax=axs[0], label='true density matrix')
        plot_Husimi_Q(self.reconstructed_dm, xgrid, pgrid, fig=fig, ax=axs[1], label='optimised density matrix')
        plt.show()

    def plot_loss_space(self):
        """
        Plots the loss functions against each other, coloured by the fidelities.
        """
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