import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from qutip import coherent, fock, thermal_dm, ket2dm, rand_dm

from qsttoolkit.data.states import cat_state, binomial_state, num_state
from qsttoolkit.data.num_state_coeffs import num_state_params, num_param_to_type
from qsttoolkit.quantum import fidelity
from qsttoolkit.plots import plot_hinton, plot_husimi_Q, plot_wigner
from qsttoolkit.utils import _deprecation_warning, _no_longer_required_warning


class StateReconstructor:
    """Class to reconstruct states from the predicted labels and key parameters. Reconstructed states are stored in the predictions_df DataFrame, along with the true states and density matrices."""
    def __init__(self):
        """Initializes the predictions_df DataFrame."""
        self.predictions_df = pd.DataFrame(columns=['true_label', 'predicted_label', 'true_state_parameter', 'predicted_state_parameter', 'restricted_predicted_state_parameter', 'true_dm', 'reconstructed_dm', 'fidelity'])

    def add_data(self, true_labels: list[str], predicted_labels: list[str], true_state_parameters: list[float], predicted_state_parameters: list[float], true_dms: list[np.ndarray]=None, true_states=None):
        """
        Supplies the true and predicted labels and state parameters, and true state density matrices, to the predictions_df DataFrame.

        Parameters
        ----------
        true_labels : list of str
            List of true labels.
        predicted_labels : list of str
            List of predicted labels.
        true_state_parameters : list of float
            List of true state parameters.
        predicted_state_parameters : list of float
            List of predicted state parameters.
        true_dms : list of np.ndarray
            List of true state density matrices.
        """
        if true_dms is None:
            if true_states is None:
                raise ValueError("missing argument: true_dms")
            else:
                _deprecation_warning('true_states', 'true_dms')
                true_dms = true_states

        self.predictions_df['true_label'] = true_labels
        self.predictions_df['predicted_label'] = predicted_labels
        self.predictions_df['true_state_parameter'] = true_state_parameters
        self.predictions_df['predicted_state_parameter'] = predicted_state_parameters
        self.predictions_df['true_dm'] = true_dms

    def restrict_parameters(self, fock_n_range: list[int,int], binomial_S_range: list[int, int]):
        """
        Restricts the predicted state parameters to be within a certain set range, depending on the predicted label, in order to enforce physicality of reconstructed states. Restricted predicted state parameters are stored in the self.predictions_df DataFrame.

        Parameters
        ----------
        fock_n_range : list of int
            List of two integers, the minimum and maximum Fock state parameter values.
        binomial_S_range : list of int
            List of two integers, the minimum and maximum binomial state parameter values.
        """
        # If the predicted label is fock or binomial, restrict the state parameter to be an integer
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: round(x['predicted_state_parameter'].real) if x['predicted_label'] in ['fock', 'binomial'] else x['predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: fock_n_range[0] if (x['predicted_label'] == 'fock') and (x['restricted_predicted_state_parameter'].real < fock_n_range[0]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: fock_n_range[1] if (x['predicted_label'] == 'fock') and (x['restricted_predicted_state_parameter'].real > fock_n_range[1]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: binomial_S_range[0] if (x['predicted_label'] == 'binomial') and (x['restricted_predicted_state_parameter'].real < binomial_S_range[0]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: binomial_S_range[1] if (x['predicted_label'] == 'binomial') and (x['restricted_predicted_state_parameter'].real > binomial_S_range[1]) else x['restricted_predicted_state_parameter'], axis=1)
        # If the predicted label is num, restrict the state parameter to be the closest of the 5 possible values
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: min(num_state_params, key=lambda y: abs(y - x['predicted_state_parameter'].real)) if x['predicted_label'] == 'num' else x['restricted_predicted_state_parameter'], axis=1)

    def reconstruct(self, Nc=None):
        """Reconstructs the states from the restricted predicted state parameters, and stores the reconstructed states and density matrices in the self.predictions_df DataFrame."""
        if Nc: _no_longer_required_warning('Nc')
        
        dim = self.predictions_df.true_dm[0].shape[0]

        for index, row in self.predictions_df.iterrows():
            if row['predicted_label'] == 'fock':
                state = fock(dim, int(row['restricted_predicted_state_parameter'].real))
                self.predictions_df.loc[index, 'reconstructed_dm'] = ket2dm(state)
            elif row['predicted_label'] == 'coherent':
                state = coherent(dim, row['restricted_predicted_state_parameter'])
                self.predictions_df.loc[index, 'reconstructed_dm'] = ket2dm(state)
            elif row['predicted_label'] == 'thermal':
                state = thermal_dm(dim, row['restricted_predicted_state_parameter'])       # Thermal initializes as a density matrix
                self.predictions_df.loc[index, 'reconstructed_dm'] = state
            elif row['predicted_label'] == 'num':
                state = num_state(num_param_to_type[row['restricted_predicted_state_parameter'].real], dim)
                self.predictions_df.loc[index, 'reconstructed_dm'] = ket2dm(state)
            elif row['predicted_label'] == 'binomial':
                S = int(row['restricted_predicted_state_parameter'].real)
                N_cap = (dim // (S + 1))-1
                if N_cap <= 2:
                    N = 2
                else:
                    N = random.randint(2, (dim // (S + 1))-1)
                mu = random.randint(0, 2)
                state = binomial_state(dim, S, N, mu)          # Binomial will be the least accurate since some parameters are guessed randomly for a certain S
                self.predictions_df.loc[index, 'reconstructed_dm'] = ket2dm(state)
            elif row['predicted_label'] == 'cat':
                state = cat_state(dim, row['restricted_predicted_state_parameter'])
                self.predictions_df.loc[index, 'reconstructed_dm'] = ket2dm(state)
            elif row['predicted_label'] == 'random':
                state = rand_dm(dim)       # Random initializes as a density matrix
                self.predictions_df.loc[index, 'reconstructed_dm'] = state

    def plot_comparison_hintons(self, state_range: list[int,int]):
        """
        Plots Hinton diagrams of the true and reconstructed density matrices for a given range of states.

        Parameters
        ----------
        state_range : list of int
            List of two integers, the minimum and maximum state indices to plot.
        """
        for i in range(state_range[0], state_range[1]):
            _, axs = plt.subplots(1, 2, figsize=(13, 5))
            plot_hinton(self.predictions_df.true_dm[i], ax=axs[0], label=f"true state {i} (type={self.predictions_df.true_label[i]}, param={np.round(self.predictions_df.true_state_parameter[i], 2)})")
            plot_hinton(self.predictions_df.reconstructed_dm[i], ax=axs[1], label=f"reconstructed state {i} (type={self.predictions_df.predicted_label[i]}, param={np.round(self.predictions_df.restricted_predicted_state_parameter[i], 2)})")
            plt.show()

    def plot_hintons(self, state_range: list[int,int]):
        """Deprecated alias for plot_comparison_hintons."""
        _deprecation_warning('pllot_hintons', 'plot_comparison_hintons')
        self.plot_comparison_hintons(state_range)

    def plot_comparison_Hintons(self, state_range: list[int,int]):
        """Deprecated alias for plot_comparison_hintons."""
        _deprecation_warning('plot_comparison_Hintons', 'plot_comparison_hintons')
        self.plot_comparison_hintons(state_range)

    def plot_comparison_husimi_Qs(self, state_range: list[int,int], xgrid: np.ndarray=None, ygrid: np.ndarray=None):
        """
        Plots Husimi Q functions of the true and reconstructed states for a given range of states.

        Parameters
        ----------
        state_range : list of int
            List of two integers, the minimum and maximum state indices to plot.
        xgrid : np.ndarray
            Grid for the real part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
        ygrid : np.ndarray
            Grid for the imaginary part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
        """
        if xgrid is None: xgrid = np.linspace(-5, 5, 100)
        if ygrid is None: ygrid = np.linspace(-5, 5, 100)
        for i in range(state_range[0], state_range[1]):
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            plot_husimi_Q(self.predictions_df.true_dm[i], xgrid, ygrid, fig, axs[0], label=f"true state {i} (type={self.predictions_df.true_label[i]}, param={np.round(self.predictions_df.true_state_parameter[i], 2)})")
            plot_husimi_Q(self.predictions_df.reconstructed_dm[i], xgrid, ygrid, fig, axs[1], label=f"reconstructed state {i} (type={self.predictions_df.predicted_label[i]}, param={np.round(self.predictions_df.restricted_predicted_state_parameter[i], 2)})")
            plt.show()

    def plot_Husimi_Qs(self, state_range: list[int,int], xgrid: np.ndarray=None, ygrid: np.ndarray=None):
        """Deprecated alias for plot_comparison_husimi_Qs."""
        _deprecation_warning('plot_Husimi_Qs', 'plot_comparison_husimi_Qs')
        self.plot_comparison_husimi_Qs(state_range, xgrid, ygrid)

    def plot_comparison_Husimi_Qs(self, state_range: list[int,int], xgrid: np.ndarray=None, ygrid: np.ndarray=None):
        """Deprecated alias for plot_comparison_husimi_Qs."""
        _deprecation_warning('plot_comparison_Husimi_Qs', 'plot_comparison_husimi_Qs')
        self.plot_comparison_husimi_Qs(state_range, xgrid, ygrid)

    def plot_comparison_wigners(self, state_range: list[int,int], xgrid: np.ndarray=None, ygrid: np.ndarray=None):
        """
        Plots Wigner functions of the true and reconstructed states for a given range of states.

        Parameters
        ----------
        state_range : list of int
            List of two integers, the minimum and maximum state indices to plot.
        xgrid : np.ndarray
            Grid for the real part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
        ygrid : np.ndarray
            Grid for the imaginary part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
        """
        if xgrid is None: xgrid = np.linspace(-5, 5, 100)
        if ygrid is None: ygrid = np.linspace(-5, 5, 100)
        for i in range(state_range[0], state_range[1]):
            fig, axs = plt.subplots(1, 2, figsize=(13, 5))
            plot_wigner(self.predictions_df.true_dm[i], xgrid, ygrid, fig, axs[0], label=f"true state {i} (type={self.predictions_df.true_label[i]}, param={np.round(self.predictions_df.true_state_parameter[i], 2)})")
            plot_wigner(self.predictions_df.reconstructed_dm[i], xgrid, ygrid, fig, axs[1], label=f"reconstructed state {i} (type={self.predictions_df.predicted_label[i]}, param={np.round(self.predictions_df.restricted_predicted_state_parameter[i], 2)})")
            plt.show()

    def calculate_fidelities(self):
        """Calculates the fidelities between the true and reconstructed states, and stores them in the self.predictions_df DataFrame 'fidelity' column."""
        self.predictions_df['fidelity'] = self.predictions_df.apply(lambda x: fidelity(x['true_dm'], x['reconstructed_dm'].full()), axis=1)
        self.predictions_df['fidelity'] = self.predictions_df.fidelity.fillna(1.0)      # 'Failed to find a square root.' indicates a perfect match

    def plot_fidelities(self, color_by_true_label: bool=False):
        """
        Plots a histogram of the fidelities between the true and reconstructed states.

        Parameters
        ----------
        color_by_true_label : bool
            If True, the fidelities are colored by the true label. Defaults to False.
        """
        _, ax = plt.subplots(figsize=(10, 7))
        if color_by_true_label:
            labels = self.predictions_df.true_label.unique()
            colors = plt.cm.get_cmap('tab10', len(labels))

            for i, label in enumerate(labels):
                data = self.predictions_df[self.predictions_df.true_label == label].fidelity
                ax.hist(data, bins=np.linspace(0, 1, 21), color=colors(i), label=label, stacked=True)
            
            ax.legend()
        else:
            ax.hist(self.predictions_df.fidelity, bins=np.linspace(0, 1, 20))

        ax.set_xlabel('Fidelity')
        ax.set_ylabel('Frequency')
        plt.show()