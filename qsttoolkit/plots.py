import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, hinton, qfunc, wigner

from qsttoolkit.utils import _deprecation_warning


##### Plotting functions #####

def plot_occupations(rho: Qobj, Nc: int, ax: plt.axes=None, label: str=None, color='#68246D', label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12, density_matrix=None) -> plt.axes:
    """
    Plots the photon number occupation probabilities for a given density matrix.
    
    Parameters
    ----------
    rho : Qobj
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
    if density_matrix is not None:
        _deprecation_warning('density_matrix', 'rho')
        rho = density_matrix
    if ax is None: _, ax = plt.subplots(figsize=(3,3))
    if type(rho) is not Qobj: rho = Qobj(rho)

    n = np.arange(0, Nc)
    n_prob = np.diag(rho.full())
    ax.bar(n, n_prob, color=color)
    ax.set_xlabel("Photon number", fontsize=axes_fontsize)
    ax.set_ylabel("Occupation probability", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Density matrix for {label}", fontsize=label_fontsize)       
    return ax

def plot_hinton(rho: Qobj, ax: plt.axes=None, label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12, density_matrix=None) -> plt.axes:
    """
    Plots the Hinton diagram of the density matrix.
    
    Parameters
    ----------
    rho : Qobj
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
    if density_matrix is not None:
        _deprecation_warning('density_matrix', 'rho')
        rho = density_matrix
    if ax is None: _, ax = plt.subplots(figsize=(3,3))
    if type(rho) is not Qobj: rho = Qobj(rho)

    hinton(rho, ax=ax, colorbar=colorbar)
    ax.set_xlabel("$|n\\rangle$", fontsize=axes_fontsize)
    ax.set_ylabel("$\\langle n|$", fontsize=axes_fontsize)
    ax.set_xticks(ax.get_xticks()[::rho.shape[0]//4 + 1])
    ax.set_yticks(ax.get_yticks()[::rho.shape[0]//4 + 1])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Density matrix for {label}", fontsize=label_fontsize)
    return ax

def plot_Hinton(density_matrix: Qobj, ax: plt.axes=None, label: str=None) -> plt.axes:
    """Deprecated alias for plot_hinton. Plots a Hinton diagram of the density matrix."""
    _deprecation_warning('plot_Hinton', 'plot_hinton')
    return plot_hinton(density_matrix, ax=ax, label=label)

def plot_husimi_Q(rho: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='hot', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12, density_matrix=None) -> plt.axes:
    """
    Plots a heatmap of the Husimi Q function of the state described by the density matrix.
    
    Parameters
    ----------
    rho : Qobj
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
    if density_matrix is not None:
        _deprecation_warning('density_matrix', 'rho')
        rho = density_matrix
    if ax is None and fig is None: fig, ax = plt.subplots(figsize=(3,3))
    if xgrid is None: xgrid = np.linspace(-5, 5, 100)
    if ygrid is None: ygrid = np.linspace(-5, 5, 100)
    if type(rho) is not Qobj: rho = Qobj(rho)
    
    Q = qfunc(rho, xgrid, ygrid)
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    im = ax.imshow(Q, extent=extent, cmap=cmap)
    if colorbar: fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_xlabel("Re($\\alpha$)", fontsize=axes_fontsize)
    ax.set_ylabel("Im($\\alpha$)", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Husimi Q function for {label}", fontsize=label_fontsize)
    return ax

def plot_Husimi_Q(density_matrix: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='hot', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """Deprecated alias for plot_husimi_Q. Plots a heatmap of the Husimi Q function of the state described by the density matrix."""
    _deprecation_warning('plot_Husimi_Q', 'plot_husimi_Q')
    return plot_husimi_Q(density_matrix, xgrid=xgrid, ygrid=ygrid, fig=fig, ax=ax, cmap=cmap, label=label, colorbar=colorbar, label_fontsize=label_fontsize, axes_fontsize=axes_fontsize, tick_fontsize=tick_fontsize)

def plot_wigner(rho: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='RdBu', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12, density_matrix=None) -> plt.axes:
    """
    Plots a heatmap of the Wigner function of the state described by the density matrix.
    
    Parameters
    ----------
    rho : Qobj
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
        Colormap to use. Defaults to 'RdBu'.
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
    if density_matrix is not None:
        _deprecation_warning('density_matrix', 'rho')
        rho = density_matrix
    if ax is None and fig is None: fig, ax = plt.subplots(figsize=(3,3))
    if xgrid is None: xgrid = np.linspace(-5, 5, 100)
    if ygrid is None: ygrid = np.linspace(-5, 5, 100)
    if type(rho) is not Qobj: rho = Qobj(rho)
    
    wig = wigner(rho, xgrid, ygrid)
    extent = [xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]]
    im = ax.imshow(wig, extent=extent, cmap=cmap)
    if colorbar: fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_xlabel("Re($\\alpha$)", fontsize=axes_fontsize)
    ax.set_ylabel("Im($\\alpha$)", fontsize=axes_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if label is not None: ax.set_title(f"Wigner function for {label}", fontsize=label_fontsize)
    return ax

def plot_Wigner(density_matrix: Qobj, xgrid: np.ndarray=None, ygrid: np.ndarray=None, fig: plt.figure=None, ax: plt.axes=None, cmap: str='RdBu', label: str=None, colorbar: bool=True, label_fontsize: float=15, axes_fontsize: float=14, tick_fontsize: float=12) -> plt.axes:
    """Deprecated alias for plot_wigner. Plots a heatmap of the Wigner function of the state described by the density matrix."""
    _deprecation_warning('plot_Wigner', 'plot_wigner')
    return plot_wigner(density_matrix, xgrid=xgrid, ygrid=ygrid, fig=fig, ax=ax, cmap=cmap, label=label, colorbar=colorbar, label_fontsize=label_fontsize, axes_fontsize=axes_fontsize, tick_fontsize=tick_fontsize)