import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from qutip import coherent, fock, thermal_dm, ket2dm, rand_dm, hinton, Qobj, qfunc

from qsttoolkit.data.states import cat_state, binomial_state, num_state
from qsttoolkit.data.num_state_coeffs import num_state_params, num_param_to_type


class StateReconstructor:
    """
    Class to reconstruct states from the predicted labels and key parameters. The reconstructed states are stored in the predictions_df DataFrame, along with the true states and density matrices.
    """
    def __init__(self):
        self.predictions_df = pd.DataFrame(columns=['true_label', 'predicted_label', 'true_state_parameter', 'predicted_state_parameter', 'restricted_predicted_state_parameter', 'true_state', 'reconstructed_state', 'reconstructed_density_matrix'])

    def add_data(self, true_labels, predicted_labels, true_state_parameters, predicted_state_parameters, true_states):
        """
        Supplies the true and predicted labels and state parameters, and true state density matrices, to the predictions_df DataFrame.

        Parameters
        ----------
        true_labels : list
            List of true labels.
        predicted_labels : list
            List of predicted labels.
        true_state_parameters : list
            List of true state parameters.
        predicted_state_parameters : list
            List of predicted state parameters.
        true_states : list
            List of true state density matrices.
        """
        self.predictions_df['true_label'] = true_labels
        self.predictions_df['predicted_label'] = predicted_labels
        self.predictions_df['true_state_parameter'] = true_state_parameters
        self.predictions_df['predicted_state_parameter'] = predicted_state_parameters
        self.predictions_df['true_state'] = true_states

    def restrict_parameters(self, fock_n_range, binomial_S_range):
        """
        Restricts the predicted state parameters to be within a certain set range, depending on the predicted label, in order to reconstruct physical states only. The restricted predicted state parameters are stored in the self.predictions_df DataFrame.

        Parameters
        ----------
        fock_n_range : list
            List of two integers, the minimum and maximum Fock state parameter values.
        binomial_S_range : list
            List of two integers, the minimum and maximum binomial state parameter values.
        """
        # If the predicted label is fock or binomial, restrict the state parameter to be an integer
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: round(x['predicted_state_parameter'].real) if x['predicted_label'] in ['fock', 'binomial'] else x['predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: 0 if (x['predicted_label'] == 'fock') and (x['restricted_predicted_state_parameter'].real < fock_n_range[0]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: 15 if (x['predicted_label'] == 'fock') and (x['restricted_predicted_state_parameter'].real > fock_n_range[1]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: 1 if (x['predicted_label'] == 'binomial') and (x['restricted_predicted_state_parameter'].real < binomial_S_range[0]) else x['restricted_predicted_state_parameter'], axis=1)
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: 10 if (x['predicted_label'] == 'binomial') and (x['restricted_predicted_state_parameter'].real > binomial_S_range[1]) else x['restricted_predicted_state_parameter'], axis=1)
        # If the predicted label is num, restrict the state parameter to be the closest of the 5 possible values
        self.predictions_df['restricted_predicted_state_parameter'] = self.predictions_df.apply(lambda x: min(num_state_params, key=lambda y: abs(y - x['predicted_state_parameter'].real)) if x['predicted_label'] == 'num' else x['restricted_predicted_state_parameter'], axis=1)

    def reconstruct(self, Nc):
        """
        Reconstructs the states from the restricted predicted state parameters, and stores the reconstructed states and density matrices in the self.predictions_df DataFrame.

        Parameters
        ----------
        Nc : int
            The Hilbert space dimensionality.
        """
        for index, row in self.predictions_df.iterrows():
            if row['predicted_label'] == 'fock':
                state = fock(Nc, int(row['restricted_predicted_state_parameter'].real))
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = ket2dm(state)
            elif row['predicted_label'] == 'coherent':
                state = coherent(Nc, row['restricted_predicted_state_parameter'])
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = ket2dm(state)
            elif row['predicted_label'] == 'thermal':
                state = thermal_dm(Nc, row['restricted_predicted_state_parameter'])       # Thermal initialises as a density matrix
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = state
            elif row['predicted_label'] == 'num':
                state = num_state(num_param_to_type[row['restricted_predicted_state_parameter'].real], Nc)
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = ket2dm(state)
            elif row['predicted_label'] == 'binomial':
                S = int(row['restricted_predicted_state_parameter'].real)
                N = random.randint(2, (Nc // (S + 1))-1)
                mu = random.randint(0, 2)
                state = binomial_state(Nc, S, N, mu)          # Binomial will be the least accurate since some parameters are guessed randomly for a certain S
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = ket2dm(state)
            elif row['predicted_label'] == 'cat':
                state = cat_state(Nc, row['restricted_predicted_state_parameter'])
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = ket2dm(state)
            elif row['predicted_label'] == 'random':
                state = rand_dm(Nc)       # Random initialises as a density matrix
                self.predictions_df.loc[index, 'reconstructed_state'] = state
                self.predictions_df.loc[index, 'reconstructed_density_matrix'] = state

    def plot_hintons(self, state_range: list[int,int]):
        """
        Plots Hinton diagrams of the true and reconstructed density matrices for a given range of states.

        Parameters
        ----------
        state_range : list
            List of two integers, the minimum and maximum state indices to plot.
        """
        import warnings
        warnings.filterwarnings("ignore")
        for i in range(state_range[0], state_range[1]):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            hinton(self.predictions_df.true_state[i], ax=axs[0])
            hinton(self.predictions_df.reconstructed_density_matrix[i], ax=axs[1])
            print(f"True state {i} (type={self.predictions_df.true_label[i]}, param={round(self.predictions_df.true_state_parameter[i], 2)}")
            print(f"Reconstructed state {i} (type={self.predictions_df.predicted_label[i]}, param={round(self.predictions_df.restricted_predicted_state_parameter[i], 2)}")
            plt.show()

    def plot_Husimi_Qs(self, state_range: list[int,int]):
        """
        Plots Husimi Q functions of the true and reconstructed states for a given range of states.

        Parameters
        ----------
        state_range : list
            List of two integers, the minimum and maximum state indices to plot.
        """
        import warnings
        warnings.filterwarnings("ignore")
        xgrid = np.linspace(-5, 5, 200)
        ygrid = np.linspace(-5, 5, 200)
        for i in range(state_range[0], state_range[1]):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            # fig.suptitle(f"State {i}")
            axs[0].imshow(qfunc(Qobj(self.predictions_df.true_state[i]), xgrid, ygrid), extent=[-5, 5, -5, 5], cmap='hot')
            axs[1].imshow(qfunc(Qobj(self.predictions_df.reconstructed_state[i]), xgrid, ygrid), extent=[-5, 5, -5, 5], cmap='hot')                # qfunc() can take either a state vector or a density matrix, but hinton() can only take density matrices. Thermal and Random states initialise as density matrices only. Custom states (Num and Binomial) must be passed into qfunc() as states
            print(f"True state {i} (type={self.predictions_df.true_label[i]}, param={round(self.predictions_df.true_state_parameter[i], 2)}")
            print(f"Reconstructed state {i} (type={self.predictions_df.predicted_label[i]}, param={round(self.predictions_df.restricted_predicted_state_parameter[i], 2)}")
            plt.show()