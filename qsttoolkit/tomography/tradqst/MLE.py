import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qsttoolkit.tomography.QST import parameterise_density_matrix, reconstruct_density_matrix, trace_constraint, positivity_constraint
from qsttoolkit.plots import plot_hinton, plot_Husimi_Q
from qsttoolkit.quantum import fidelity


def log_likelihood(params: np.array, dim: int, projective_operators: np.array, frequency_data: np.array, L1_reg=0) -> float:
    """
    Computes the negative log-likelihood of the data given the density matrix.

    Parameters
    ----------
    params : np.array
        The parameters of the density matrix.
    dim : int
        The Hilbert space dimensionality.
    projective_operators : np.array
        The projective operators corresponding to the measurement outcomes.
    frequency_data : np.array
        The frequency of each measurement outcome.
    L1_reg : float
        The L1 regularisation parameter. Default is 0.

    Returns
    -------
    float
        The negative log-likelihood of the data given the density matrix.
    """
    rho = reconstruct_density_matrix(params, dim)
    log_likelihood = 0
    # Insert error handling to ensure rho and E_K are the same dimension
    for E_k, f_k in zip(projective_operators, frequency_data):
        p_k = np.trace(rho @ E_k).real          # Probability of outcome k
        if p_k > 0:
            log_likelihood += f_k * np.log(p_k)
    
    return -log_likelihood + L1_reg * np.sum(np.abs(params))


class MLEQuantumStateTomography:
    """
    A class for performing maximum likelihood estimation quantum state tomography.

    Attributes
    ----------
    dim : int
        The Hilbert space dimensionality.
    constraints : list
        The constraint functions for the optimisation.
    initial_dm : np.array
        The initial density matrix guess.
    initial_params : np.array
        The initial parameters of the density matrix guess.
    history : list
        The history of the optimisation.
    cost_values : list
        The cost values over the iterations.
    result : scipy.optimize.OptimizeResult
        The result of the optimisation.
    reconstructed_dm : np.array
        The reconstructed density matrix.
    fidelities : list
        The fidelities between the true and reconstructed density matrices.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.constraints = [{'type': 'eq', 'fun': trace_constraint, 'args': (dim,)},
                            {'type': 'ineq', 'fun': positivity_constraint, 'args': (dim,)}]

    def reconstruct(self, measurement_data: np.array, measurement_operators: np.array, initial_dm: np.array, method='SLSQP', L1_reg=0, verbose=True):
        """
        Fits the density matrix to the data using the maximum likelihood estimator.

        Parameters
        ----------
        measurement_data : np.array
            The frequency of each measurement outcome.
        measurement_operators : np.array
            The projective operators corresponding to the measurement outcomes.
        initial_dm : np.array
            The initial density matrix guess.
        method : str
            The optimisation method. Default is 'SLSQP'.
        L1_reg : float
            The L1 regularisation parameter. Default is 0.
        verbose : bool
            Whether to print the progress of the optimisation. Default is True.
        """
        self.initial_dm = initial_dm
        self.initial_params = parameterise_density_matrix(initial_dm)
        if verbose:
            self.options = {'disp': True}
        else:
            self.options = {'disp': False}

        self.history = []
        self.cost_values = []

        def record_optimisation_progress(x):
            """Records the progress of the optimisation."""
            self.history.append(x)
            log_likelihood_value = log_likelihood(x, self.dim, measurement_operators, measurement_data)
            self.cost_values.append(log_likelihood_value)
            if verbose: print(f"Current cost: {log_likelihood_value}")

        self.result = minimize(log_likelihood,
                               self.initial_params,
                               args=(self.dim, measurement_operators, measurement_data, L1_reg),
                               method=method,
                               constraints=self.constraints,
                               callback=record_optimisation_progress,
                               options=self.options)

        self.reconstructed_dm = reconstruct_density_matrix(self.result.x, self.dim)
        if verbose: print('Optimisation terminated successfully:', self.result.success)

    def plot_cost_values(self):
        """
        Plots the cost function over the iterations of the optimisation.
        """
        plt.figure(figsize=(5, 4))
        plt.plot(self.cost_values)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost function over iterations')
        plt.show()

    def plot_fidelities(self, true_dm: np.array):
        """
        Plots the fidelity between the true and reconstructed density matrices over the iterations.

        Parameters
        ----------
        true_dm : np.array
            The true density matrix.
        """
        fidelities = []
        for x in self.history:
            fidelities.append(fidelity(true_dm.full(), reconstruct_density_matrix(x, self.dim)))
        self.fidelities = fidelities

        plt.figure(figsize=(5, 4))
        plt.plot(self.fidelities)
        plt.ylim(0,1)
        plt.xlabel('Iteration')
        plt.ylabel('Fidelity')
        plt.title('Fidelity over iterations')
        plt.show()

    def plot_comparison_hintons(self, true_dm: np.array):
        """
        Plots the Hinton diagrams of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.array
            The true density matrix.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_hinton(true_dm, ax=axs[0], label='true density matrix')
        plot_hinton(self.reconstructed_dm, ax=axs[1], label='reconstructed density matrix')
        plt.show()

    def plot_comparison_Husimi_Qs(self, true_dm: np.array, xgrid: np.array, pgrid: np.array):
        """
        Plots the Husimi-Q functions of the true and reconstructed density matrices.

        Parameters
        ----------
        true_dm : np.array
            The true density matrix.
        xgrid : np.array
            The phase space x grid for the Husimi-Q function.
        pgrid : np.array
            The phase space p grid for the Husimi-Q function.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_Husimi_Q(true_dm, xgrid, pgrid, fig=fig, ax=axs[0], label='true density matrix')
        plot_Husimi_Q(self.reconstructed_dm, xgrid, pgrid, fig=fig, ax=axs[1], label='optimised density matrix')
        plt.show()

    def plot_intermediate_hintons(self):
        """
        Plots the Hinton diagrams of the density matrices at five stages of the optimisation.
        """
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        plot_hinton(self.initial_dm, ax=axs[0], label='initial guess')
        plot_hinton(reconstruct_density_matrix(self.history[len(self.history) // 4], self.dim), ax=axs[1], label=f"iteration {len(self.history) // 4}")
        plot_hinton(reconstruct_density_matrix(self.history[len(self.history) // 2], self.dim), ax=axs[2], label=f"iteration {len(self.history) // 2}")
        plot_hinton(reconstruct_density_matrix(self.history[3 * len(self.history) // 4], self.dim), ax=axs[3], label=f"iteration {3 * len(self.history) // 4}")
        plot_hinton(self.reconstructed_dm, ax=axs[4], label='result')
        plt.show()

    def plot_intermediate_Husimi_Qs(self, xgrid: np.array, pgrid: np.array):
        """
        Plots the Husimi-Q functions of the density matrices at five stages of the optimisation.

        Parameters
        ----------
        xgrid : np.array
            The phase space x grid for the Husimi-Q function.
        pgrid : np.array
            The phase space p grid for the Husimi-Q function.
        """
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        plot_Husimi_Q(reconstruct_density_matrix(self.history[0], self.dim), xgrid, pgrid, fig=fig, ax=axs[0], label='initial guess')
        plot_Husimi_Q(reconstruct_density_matrix(self.history[len(self.history) // 4], self.dim), xgrid, pgrid, fig=fig, ax=axs[1], label=f"iteration {len(self.history) // 4}")
        plot_Husimi_Q(reconstruct_density_matrix(self.history[len(self.history) // 2], self.dim), xgrid, pgrid, fig=fig, ax=axs[2], label=f"iteration {len(self.history) // 2}")
        plot_Husimi_Q(reconstruct_density_matrix(self.history[3 * len(self.history) // 4], self.dim), xgrid, pgrid, fig=fig, ax=axs[3], label=f"iteration {3 * len(self.history) // 4}")
        plot_Husimi_Q(self.reconstructed_dm, xgrid, pgrid, fig=fig, ax=axs[4], label='result')
        plt.show()