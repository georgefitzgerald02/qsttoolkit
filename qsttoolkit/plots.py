import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, hinton, qfunc, wigner
import warnings


##### Plotting functions #####

def plot_occupations(density_matrix: Qobj, Nc: int, ax: plt.axes=None, label: str=None, color='#68246D', label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """
    Plots the photon number occupation probabilities for a given density matrix.
    
    Parameters
    ----------
    density_matrix : Qobj
        Density matrix to be plotted.
    Nc : int
        Hilbert space cutoff.
    ax : plt.axes
        Axes object to plot on. If None, a new figure is created.
    label : str
        Label for the plot. Defaults to None.
    color : str
        Color of the bars. Defaults to '#68246D' (Palatinate purple).
    label_fontsize : float
        Fontsize of the title. Defaults to 15.
    axes_fontsize : float
        Fontsize of the axes labels. Defaults to 14.
    tick_fontsize : float
        Fontsize of the tick labels. Defaults to 12.

    Returns
    -------
    plt.axes
        Axes object containing the plot.
    """
    if ax is None: _, ax = plt.subplots(figsize=(3,3))
    if type(density_matrix) is not Qobj: density_matrix = Qobj(density_matrix)

    n = np.arange(0, Nc)
    n_prob = np.diag(density_matrix.full())
    ax.bar(n, n_prob, color=color)
    ax.set_xlabel("Photon number", fontsize=axes_fontsize)
    ax.set_ylabel("Occupation probability", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Density matrix for {label}", fontsize=label_fontsize)       
    return ax

def plot_Hinton(density_matrix: Qobj, ax: plt.axes=None, label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """
    Plots the Hinton diagram of the density matrix.
    
    Parameters
    ----------
    density_matrix : Qobj
        Density matrix to be plotted.
    ax : plt.axes
        Axes object to plot on. If None, a new figure is created.
    label : str
        Label for the plot. Defaults to None.
    colorbar : bool
        Whether to include a colorbar. Defaults to True.
    label_fontsize : float
        Fontsize of the title. Defaults to 15.
    axes_fontsize : float
        Fontsize of the axes labels. Defaults to 14.
    tick_fontsize : float
        Fontsize of the tick labels. Defaults to 12.

    Returns
    -------
    plt.axes
        Axes object containing the plot.
    """
    if ax is None: _, ax = plt.subplots(figsize=(3,3))
    if type(density_matrix) is not Qobj: density_matrix = Qobj(density_matrix)

    hinton(density_matrix, ax=ax, colorbar=colorbar)
    ax.set_xlabel("$|n\\rangle$", fontsize=axes_fontsize)
    ax.set_ylabel("$\\langle n|$", fontsize=axes_fontsize)
    ax.set_xticks(ax.get_xticks()[::density_matrix.shape[0]//4 + 1])
    ax.set_yticks(ax.get_yticks()[::density_matrix.shape[0]//4 + 1])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Density matrix for {label}", fontsize=label_fontsize)
    return ax

def plot_hinton(density_matrix: Qobj, ax: plt.axes=None, label: str=None) -> plt.axes:
    """Deprecated alias for plot_Hinton. Plots a Hinton diagram of the density matrix."""
    warnings.warn("plot_hinton is deprecated and will be removed in a future version. Please use plot_Hinton instead.", DeprecationWarning, stacklevel=2)
    return plot_Hinton(density_matrix, ax=ax, label=label)

def plot_Husimi_Q(density_matrix: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='hot', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """
    Plots a heatmap of the Husimi Q function of the state described by the density matrix.
    
    Parameters
    ----------
    density_matrix : Qobj
        Density matrix to be plotted.
    xgrid : np.ndarray
        Grid for the real part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
    ygrid : np.ndarray
        Grid for the imaginary part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
    fig : plt.figure
        Figure object to plot on. If None, a new figure is created.
    ax : plt.axes
        Axes object to plot on. If None, a new figure is created.
    cmap : str
        Colormap to use. Defaults to 'hot'.
    label : str
        Label for the plot. Defaults to None.
    colorbar : bool
        Whether to include a colorbar. Defaults to True.
    label_fontsize : float
        Fontsize of the title. Defaults to 15.
    axes_fontsize : float
        Fontsize of the axes labels. Defaults to 14.
    tick_fontsize : float
        Fontsize of the tick labels. Defaults to 12.

    Returns
    -------
    plt.axes
        Axes object containing the plot.
    """
    if ax is None and fig is None: fig, ax = plt.subplots(figsize=(3,3))
    if xgrid is None: xgrid = np.linspace(-5, 5, 100)
    if ygrid is None: ygrid = np.linspace(-5, 5, 100)
    if type(density_matrix) is not Qobj: density_matrix = Qobj(density_matrix)
    
    Q = qfunc(density_matrix, xgrid, ygrid)
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    im = ax.imshow(Q, extent=extent, cmap=cmap)
    if colorbar: fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_xlabel("Re($\\beta$)", fontsize=axes_fontsize)
    ax.set_ylabel("Im($\\beta$)", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Husimi Q function for {label}", fontsize=label_fontsize)
    return ax

def plot_Wigner(density_matrix: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='hot', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """
    Plots a heatmap of the Wigner function of the state described by the density matrix.
    
    Parameters
    ----------
    density_matrix : Qobj
        Density matrix to be plotted.
    xgrid : np.ndarray
        Grid for the real part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
    ygrid : np.ndarray
        Grid for the imaginary part of the coherent state parameter. Defaults to np.linspace(-5, 5, 100).
    fig : plt.figure
        Figure object to plot on. If None, a new figure is created.
    ax : plt.axes
        Axes object to plot on. If None, a new figure is created.
    cmap : str
        Colormap to use. Defaults to 'hot'.
    label : str
        Label for the plot. Defaults to None.
    colorbar : bool
        Whether to include a colorbar. Defaults to True.
    label_fontsize : float
        Fontsize of the title. Defaults to 15.
    axes_fontsize : float
        Fontsize of the axes labels. Defaults to 14.
    tick_fontsize : float
        Fontsize of the tick labels. Defaults to 12.

    Returns
    -------
    plt.axes
        Axes object containing the plot.
    """
    if ax is None and fig is None: fig, ax = plt.subplots(figsize=(3,3))
    if xgrid is None: xgrid = np.linspace(-5, 5, 100)
    if ygrid is None: ygrid = np.linspace(-5, 5, 100)
    if type(density_matrix) is not Qobj: density_matrix = Qobj(density_matrix)
    
    wig = wigner(density_matrix, xgrid, ygrid)
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    im = ax.imshow(wig, extent=extent, cmap=cmap)
    if colorbar: fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_xlabel("Re($\\beta$)", fontsize=axes_fontsize)
    ax.set_ylabel("Im($\\beta$)", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Wigner function for {label}", fontsize=label_fontsize)
    return ax