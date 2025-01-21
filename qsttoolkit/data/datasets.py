import numpy as np
import pandas as pd
from qutip import qfunc

from qsttoolkit.data.state_batches import FockStates, CoherentStates, ThermalStates, NumStates, BinomialStates, CatStates, RandomStates
from qsttoolkit.data.noise import mixed_state_noise, apply_measurement_noise


def optical_state_dataset(dim: int = 32, latent_dim: int = 32) -> pd.DataFrame:
    """
    Generates a standardised dataset of optical quantum states with added noise for training machine learning quantum state tomography models.
    
    Parameters
    ----------
    dim : int, default=32
        The dimension of the Hilbert space.
    latent_dim : int, default=32
        The number of points in the x and p grids for the Q-function.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the labels, density matrices, Husimi-Q functions, and state parameters of the states.
    """
    fock_batch = FockStates(n_states = 1000,
                            N = dim,
                            n_range = [0, dim])
    coherent_batch = CoherentStates(n_states = 1000,
                                    N = dim,
                                    alpha_magnitude_range = [1e-6, 3])
    thermal_batch = ThermalStates(n_states = 1000,
                                  N = dim,
                                  nbar_range = [0, dim])
    num_batch = NumStates(n_states = 1000,
                          N = dim,
                          types = ['17', 'M', 'P', 'P2', 'M2'])
    binomial_batch = BinomialStates(n_states = 1000,
                                    N = dim,
                                    S_range = [1, 10],
                                    mu_range = [0, 2])
    cat_batch = CatStates(n_states = 1000,
                          N = dim,
                          alpha_magnitude_range = [0, 10])
    random_batch = RandomStates(n_states = 1000,
                                N = dim)
    print('States generated')

    mixed_state_noise_noise_level = 0.3
    affine_theta = 30
    affine_x = 0.1
    affine_y = 0.1
    additive_Gaussian_stddev = 0.01
    salt_and_pepper_prob = 0.1

    # Create phase space grid
    xgrid = np.linspace(-5, 5, latent_dim)
    pgrid = np.linspace(-5, 5, latent_dim)

    fock_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    coherent_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    thermal_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    num_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    binomial_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    cat_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    random_data = pd.DataFrame(columns=['label', 'density_matrix', 'Husimi-Q_function', 'state_parameter'])
    print("DataFrames initialised")

    fock_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in fock_batch.density_matrices()]
    fock_data['label'] = ['fock']*len(fock_densities)
    fock_data['density_matrix'] = [dm.full() for dm in fock_densities]
    fock_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in fock_densities]
    fock_data['state_parameter'] = fock_batch.params
    print("Fock data generated")

    coherent_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in coherent_batch.density_matrices()]
    coherent_data['label'] = ['coherent']*len(coherent_densities)
    coherent_data['density_matrix'] = [dm.full() for dm in coherent_densities]
    coherent_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in coherent_densities]
    coherent_data['state_parameter'] = coherent_batch.params
    print("Coherent data generated")

    thermal_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in thermal_batch.density_matrices()]
    thermal_data['label'] = ['thermal']*len(thermal_densities)
    thermal_data['density_matrix'] = [dm.full() for dm in thermal_densities]
    thermal_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in thermal_densities]
    thermal_data['state_parameter'] = thermal_batch.params
    print("Thermal data generated")

    num_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in num_batch.density_matrices()]
    num_data['label'] = ['num']*len(num_densities)
    num_data['density_matrix'] = [dm.full() for dm in num_densities]
    num_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in num_densities]
    num_data['state_parameter'] = num_batch.params
    print("Num data generated")

    binomial_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in binomial_batch.density_matrices()]
    binomial_data['label'] = ['binomial']*len(binomial_densities)
    binomial_data['density_matrix'] = [dm.full() for dm in binomial_densities]
    binomial_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in binomial_densities]
    binomial_data['state_parameter'] = binomial_batch.params
    print("Binomial data generated")

    cat_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in cat_batch.density_matrices()]
    cat_data['label'] = ['cat']*len(cat_densities)
    cat_data['density_matrix'] = [dm.full() for dm in cat_densities]
    cat_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in cat_densities]
    cat_data['state_parameter'] = cat_batch.params
    print("Cat data generated")

    random_densities = [mixed_state_noise(dm, mixed_state_noise_noise_level) for dm in random_batch.density_matrices()]
    random_data['label'] = ['random']*len(random_densities)
    random_data['density_matrix'] = [dm.full() for dm in random_densities]
    random_data['Husimi-Q_function'] = [apply_measurement_noise(qfunc(dm, xgrid, pgrid), affine_theta, affine_x, affine_y, additive_Gaussian_stddev, salt_and_pepper_prob) for dm in random_densities]
    random_data['state_parameter'] = random_batch.params
    print("Random data generated")

    data = pd.concat([fock_data, coherent_data, thermal_data, num_data, binomial_data, cat_data, random_data])
    data = data.sample(frac=1).reset_index(drop=True)
    print("Dataset generated")
    return data