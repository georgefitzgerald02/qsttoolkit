import numpy as np
import matplotlib.pyplot as plt

from qutip import Qobj, hinton, qfunc, wigner


def plot_occupations(density_matrix: np.ndarray, Nc: int, ax=None, label: str=None, color: str='green') -> plt.axes:
    """
    Plots the occupation probabilities of the photon number states for a given density matrix.
    
    Parameters
    ----------
    density_matrix : np.ndarray
        The density matrix to be plotted.
    Nc : int
        The Hilbert space cutoff.
    ax : plt.axes, optional
        The axes object to plot on. If None, a new figure is created.
    label : str, optional
        A label for the plot.
    color : str, optional
        The color of the bars. Default is 'green'.

    Returns
    -------
    plt.axes
        The axes object containing the plot.
    """
    if ax is None: fig, ax = plt.subplots(figsize=(3,3))
    n = np.arange(0, Nc)
    n_prob = np.diag(density_matrix)
    ax.bar(n, n_prob, color=color)
    ax.set_xlabel("Photon number")
    ax.set_ylabel("Occupation probability")
    if label is not None: ax.set_title(f"Density matrix for state {label}")
    return ax

def plot_hinton(density_matrix: np.ndarray, ax=None, label: str=None) -> plt.axes:
    """
    Plots the Hinton diagram of the density matrix.
    
    Parameters
    ----------
    density_matrix : np.ndarray
        The density matrix to be plotted.
    ax : plt.axes, optional
        The axes object to plot on. If None, a new figure is created.
    label : str, optional
        A label for the plot.

    Returns
    -------
    plt.axes
        The axes object containing the plot.
    """
    if ax is None: fig, ax = plt.subplots(figsize=(3,3))
    hinton(Qobj(density_matrix), ax=ax)
    ax.set_xlabel("$|n\\rangle$")
    ax.set_ylabel("$\\langle n|$")
    ax.set_xticks(ax.get_xticks()[::density_matrix.shape[0]//4 + 1])
    ax.set_yticks(ax.get_yticks()[::density_matrix.shape[0]//4 + 1])
    if label is not None: ax.set_title(f"Density matrix for state {label}")
    return ax

def plot_Husimi_Q(density_matrix: np.ndarray, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig=None, ax=None, cmap: str='hot', label: str=None) -> plt.axes:
    """
    Plots a heatmap of the Husimi-Q function of the state described by the density matrix.
    
    Parameters
    ----------
    density_matrix : np.ndarray
        The density matrix to be plotted.
    xgrid : np.ndarray, optional
        The grid for the real part of the coherent state parameter. Default is np.linspace(-5, 5, 100).
    ygrid : np.ndarray, optional
        The grid for the imaginary part of the coherent state parameter. Default is np.linspace(-5, 5, 100).
    fig : plt.figure, optional
        The figure object to plot on. If None, a new figure is created.
    ax : plt.axes, optional
        The axes object to plot on. If None, a new figure is created.
    cmap : str, optional
        The colormap to use. Default is 'hot'.
    label : str, optional
        A label for the plot.

    Returns
    -------
    plt.axes
        The axes object containing the plot.
    """
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(3,3))
    if xgrid is None: xgrid = np.linspace(-5, 5, 100)
    if ygrid is None: ygrid = np.linspace(-5, 5, 100)
    Q = qfunc(Qobj(density_matrix), xgrid, ygrid)
    im = ax.imshow(Q, extent=[-5, 5, -5, 5], cmap=cmap)
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_xlabel("Re($\\beta$)")
    ax.set_ylabel("Im($\\beta$)")
    if label is not None: ax.set_title(f"Husimi-Q function for state {label}")
    return ax