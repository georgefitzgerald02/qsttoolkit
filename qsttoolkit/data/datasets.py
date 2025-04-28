import numpy as np
import pandas as pd
from qutip import qfunc
import warnings

from qsttoolkit.data.state_batches import FockStates, CoherentStates, ThermalStates, NumStates, BinomialStates, CatStates, GKPStates, RandomStates
from qsttoolkit.data.noise import mixed_state_noise, apply_measurement_noise
from qsttoolkit.utils import _deprecation_warning


def optical_state_dataset(dim: int, data_dim: int=None, state_numbers: list=[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], mixed_state_noise_level: float=0.1, affine_theta: float=20.0, affine_x: float=0.1, affine_y: float=0.1, additive_Gaussian_stddev: float=0.01, amplification_ntherm: float=1.0, pepper_p: float=0.01, salt_p: float=0.0, latent_dim=None, mixed_state_noise_noise_level=None, Gaussian_conv_ntherm=None) -> pd.DataFrame:
    """
    Generates a standardized dataset of optical quantum states with added noise for training machine learning quantum state discrimination and tomography models.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the Hilbert space.
    data_dim : int
        Number of points in the X and P grids for the Q-function.
    state_numbers : list[int]
        List of 8 integers specifying the number of states to generate for each state type in the dataset. order is [Fock, Coherent, Thermal, Num, Binomial, Cat, GKP, Random]. Defalts to [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000].
    mixed_state_noise_level : float
        Random state coefficient for the mixed state noise. Defaults to 0.1.
    affine_theta : float
        Maximum rotation angle in degrees. Defaults to 20.0.
    affine_x : float
        Maximum translation in the x direction. Defaults to 0.1.
    affine_y : float
        Maximum translation in the y direction. Defaults to 0.1.
    additive_Gaussian_stddev : float
        Standard deviation of the Gaussian distribution from which additive noise is sampled. Defaults to 0.01.
    amplification_ntherm : float
        Mean photon number, equal to the variance of the Gaussian convolution kernel. Defaults to 1.0.
    salt_p : float
        Proportion of pixels to set to 1. Defaults to 0.0.
    pepper_p : float
        Proportion of pixels to set to 0. Defaults to 0.01.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the labels, density matrices, Husimi Q functions, and state parameters of the states.
    """
    if not data_dim:
        if latent_dim:
            _deprecation_warning('latent_dim', 'data_dim')
            data_dim = latent_dim
        else:
            raise ValueError("data_dim must be specified.")

    if mixed_state_noise_noise_level:
        _deprecation_warning('mixed_state_noise_noise_level', 'mixed_state_noise_level')
        mixed_state_noise_level = mixed_state_noise_noise_level

    if Gaussian_conv_ntherm:
        _deprecation_warning('Gaussian_conv_ntherm', 'amplification_ntherm')
        amplification_ntherm = Gaussian_conv_ntherm

    if not isinstance(dim, int): raise ValueError("dim must be an integer.")
    if not isinstance(data_dim, int): raise ValueError("data_dim must be an integer.")
    if not (isinstance(state_numbers, list) and len(state_numbers) == 8 and all(isinstance(n, int) and n >= 0 for n in state_numbers)): raise ValueError("state_numbers must be a list of 8 positive integers.")


    fock_batch = FockStates(n_states = state_numbers[0],
                            dim = dim,
                            n_range = [0, dim])
    print("Fock states generated")
    coherent_batch = CoherentStates(n_states = state_numbers[1],
                                    dim = dim,
                                    alpha_magnitude_range = [1e-6, 3])
    print("Coherent states generated")
    thermal_batch = ThermalStates(n_states = state_numbers[2],
                                  dim = dim,
                                  nbar_range = [0, dim])
    print("Thermal states generated")
    num_batch = NumStates(n_states = state_numbers[3],
                          dim = dim,
                          types = ['17', 'M', 'P', 'P2', 'M2'])
    print("Num states generated")
    binomial_batch = BinomialStates(n_states = state_numbers[4],
                                    dim = dim,
                                    S_range = [1, 10],
                                    mu_range = [0, 2])
    print("Binomial states generated")
    cat_batch = CatStates(n_states = state_numbers[5],
                          dim = dim,
                          alpha_magnitude_range = [0, 10])
    print("Cat states generated")
    gkp_batch = GKPStates(n_states = state_numbers[6],
                          dim = dim,
                          n1_range = [-20, 20],
                            n2_range = [-20, 20],
                            delta_range=[0.2, 0.5],
                            mu_range=[0, 2])
    print("GKP states generated")
    random_batch = RandomStates(n_states = state_numbers[7],
                                dim = dim)
    print("Random states generated")

    # Create phase space grid
    xgrid = np.linspace(-5, 5, data_dim)
    pgrid = np.linspace(-5, 5, data_dim)

    fock_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    coherent_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    thermal_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    num_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    binomial_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    cat_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    gkp_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    random_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    warnings.warn("The column 'Husimi-Q' will be renamed to 'Husimi_Q' in the next major update.", FutureWarning, stacklevel=2)
    print("DataFrames initialized")

    fock_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in fock_batch.density_matrices()]
    fock_data['label'] = ['fock']*len(fock_densities)
    fock_data['density_matrix'] = [dm.full() for dm in fock_densities]
    fock_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in fock_densities]
    fock_data['state_parameter'] = fock_batch.params
    print("Fock data generated")

    coherent_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in coherent_batch.density_matrices()]
    coherent_data['label'] = ['coherent']*len(coherent_densities)
    coherent_data['density_matrix'] = [dm.full() for dm in coherent_densities]
    coherent_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in coherent_densities]
    coherent_data['state_parameter'] = coherent_batch.params
    print("Coherent data generated")

    thermal_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in thermal_batch.density_matrices()]
    thermal_data['label'] = ['thermal']*len(thermal_densities)
    thermal_data['density_matrix'] = [dm.full() for dm in thermal_densities]
    thermal_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in thermal_densities]
    thermal_data['state_parameter'] = thermal_batch.params
    print("Thermal data generated")

    num_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in num_batch.density_matrices()]
    num_data['label'] = ['num']*len(num_densities)
    num_data['density_matrix'] = [dm.full() for dm in num_densities]
    num_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in num_densities]
    num_data['state_parameter'] = num_batch.params
    print("Num data generated")

    binomial_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in binomial_batch.density_matrices()]
    binomial_data['label'] = ['binomial']*len(binomial_densities)
    binomial_data['density_matrix'] = [dm.full() for dm in binomial_densities]
    binomial_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in binomial_densities]
    binomial_data['state_parameter'] = binomial_batch.params
    print("Binomial data generated")

    cat_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in cat_batch.density_matrices()]
    cat_data['label'] = ['cat']*len(cat_densities)
    cat_data['density_matrix'] = [dm.full() for dm in cat_densities]
    cat_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in cat_densities]
    cat_data['state_parameter'] = cat_batch.params
    print("Cat data generated")

    gkp_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in gkp_batch.density_matrices()]
    gkp_data['label'] = ['gkp']*len(gkp_densities)
    gkp_data['density_matrix'] = [dm.full() for dm in gkp_densities]
    gkp_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in gkp_densities]
    gkp_data['state_parameter'] = gkp_batch.params
    print("GKP data generated")

    random_densities = [mixed_state_noise(dm, mixed_state_noise_level) for dm in random_batch.density_matrices()]
    random_data['label'] = ['random']*len(random_densities)
    random_data['density_matrix'] = [dm.full() for dm in random_densities]
    random_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta=affine_theta, affine_x=affine_x, affine_y=affine_y, amplification_ntherm=amplification_ntherm, additive_Gaussian_stddev=additive_Gaussian_stddev, pepper_p=pepper_p, salt_p=salt_p) for dm in random_densities]
    random_data['state_parameter'] = random_batch.params
    print("Random data generated")

    data = pd.concat([fock_data, coherent_data, thermal_data, num_data, binomial_data, cat_data, gkp_data, random_data])
    data = data.sample(frac=1).reset_index(drop=True)
    print("Dataset generated")
    return data