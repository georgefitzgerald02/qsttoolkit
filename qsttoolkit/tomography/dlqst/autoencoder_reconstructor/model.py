import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from qsttoolkit.tomography.QST import QuantumStateTomography, reconstruct_density_matrix
from qsttoolkit.tomography.dlqst.GAN_reconstructor.architecture import build_generator
from qsttoolkit.tomography.dlqst.autoencoder_reconstructor.train import train
from qsttoolkit.plots import plot_hinton, plot_Husimi_Q
from qsttoolkit.utils import _subplot_number, _subplot_figsize


class AutoencoderQuantumStateTomography(QuantumStateTomography):
    """
    A class for training and evaluating an autoencoder for quantum state tomography. A single neural network is trained to minimise the negative log-likelihood of the measurement data.

    Attributes
    ----------
    dim : int
        The Hilbert space dimensionality.
    model : tf.keras.Model
        The encoding network.
    losses : list
        Loss function values over the optimisation.
    progress_saves : list
        Density matrix progress saves.
    fidelities : list
        Fidelities between the true and reconstructed density matrices over epochs.
    reconstructed_dm : np.ndarray
        The reconstructed density matrix.
    """
    def __init__(self, data_dim: int):
        super().__init__()
        self.model = build_generator(data_vector_input_shape=(data_dim,))

    def reconstruct(self, measurement_data: list, measurement_operators: list, epochs=100, optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002), verbose_interval=None, num_progress_saves=None, true_dm=None):
        """
        Trains the GAN to reconstruct the density matrix from measurement data.

        Parameters
        ----------
        measurement_data : list of np.ndarray
            The frequency of each measurement outcome.
        measurement_operators : list of Qobj
            The projective operators corresponding to the measurement outcomes.
        epochs : int
            The number of epochs to train for. Defaults to 100.
        verbose_interval : int
            The interval at which to print progress updates. Defaults to None.
        num_progress_saves : int
            The number of intermediate progress saves to make. Defaults to None.
        true_dm : np.ndarray
            The true density matrix used for calculating fidelities. Defaults to None.
        """
        self.losses, self.progress_saves, self.fidelities = train(self.model,
                                                                  measurement_data,
                                                                  measurement_operators,
                                                                  epochs=epochs,
                                                                  optimiser=optimiser,
                                                                  verbose_interval=verbose_interval,
                                                                  num_progress_saves=num_progress_saves,
                                                                  true_dm=true_dm)                        

        self.reconstructed_dm = reconstruct_density_matrix(self.model(measurement_data)).numpy()
        if verbose_interval: print("Reconstruction complete.")

    def plot_comparison_hintons(self, true_dms: list):
        """
        Plots Hinton diagrams of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.ndarray
            The true density matrix.
        """
        fig, axs = plt.subplots(len(true_dms), 2, figsize=(10, 5*len(true_dms)))
        axs = np.array(axs).flatten()
        for i, dm in enumerate(true_dms):
            plot_hinton(dm, ax=axs[i], label='true density matrix')
            plot_hinton(self.reconstructed_dm[i], ax=axs[i+1], label='reconstructed density matrix')
        plt.show()

    def plot_comparison_Husimi_Qs(self, true_dms: list, xgrid: np.ndarray, pgrid: np.ndarray):
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
        fig, axs = plt.subplots(len(true_dms), 2, figsize=(10, 5*len(true_dms)))
        axs = np.array(axs).flatten()
        for i, dm in enumerate(true_dms):
            plot_Husimi_Q(dm, xgrid, pgrid, fig=fig, ax=axs[i], label='true density matrix')
            plot_Husimi_Q(self.reconstructed_dm[i], xgrid, pgrid, fig=fig, ax=axs[i+1], label='reconstructed density matrix')
        plt.show()

    def plot_intermediate_hintons(self):
        """
        Plots Hinton diagrams of the density matrices in the progress_saves attribute.
        """
        for run in range(self.progress_saves[0].shape[0]):
            subplot_number = _subplot_number(len(self.progress_saves))
            fig, axs = plt.subplots(subplot_number[0], subplot_number[1], figsize=_subplot_figsize(len(self.progress_saves)), squeeze=False)
            axs = np.array(axs).flatten()
            for i, save in enumerate(self.progress_saves):
                plot_hinton(save[run], ax=axs[i], label=f"save {i}")
                plt.suptitle(f"Reconstruction of matrix at index {run}:")
            plt.show()

    def plot_intermediate_Husimi_Qs(self, xgrid: np.ndarray, pgrid: np.ndarray):
        """
        Plots the Husimi-Q functions of the density matrices in the progress_saves attribute.

        Parameters
        ----------
        xgrid : np.ndarray
            The phase space x grid for the Husimi-Q function.
        pgrid : np.ndarray
            The phase space p grid for the Husimi-Q function.
        """
        for run in range(self.progress_saves[0].shape[0]):
            subplot_number = _subplot_number(len(self.progress_saves))
            fig, axs = plt.subplots(subplot_number[0], subplot_number[1], figsize=_subplot_figsize(len(self.progress_saves)))
            axs = axs.flatten()
            for i, save in enumerate(self.progress_saves):
                plot_Husimi_Q(save[run], xgrid, pgrid, fig=fig, ax=axs[i], label=f"save {i}")
            plt.suptitle(f"Reconstruction of matrix at index {run}:")
            plt.show()