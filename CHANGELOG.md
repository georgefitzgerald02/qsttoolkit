# Changelog

The format of this Changelog is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). QSTToolkit adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Please see the official [documentation](https://qsttoolkit.readthedocs.io/en/latest/) for full details of all features.

## Contents
- [1.2.0](#v120---2025-04-28)
- [1.1.1](#v111---2025-04-14)
- [1.1.0](#v110---2025-03-19)
- [1.0.0](#v100---2025-01-22)

## v1.2.0 - 2025-04-28

A variety of new features primed towards customisable, multi-purpose tomography, including new loss functions, measurement regimes, noise sources qubit states (expanding on those provided by QuTiP) the introduction of the `CustomQuantumStateTomography` sandbox environment for combining features of existing tomography methods to create new approaches.

### Added

- The `CustomQuantumStateTomography` class, allowing a user to define and optimize a custom training step for tomography using existing QSTToolkit functions (built in TensorFlow). This is accompanied by a new `Custom_tomography_demo.ipynb` example notebook in the GitHub repo.
- Mean absolute error, mean squared error, KL divergence and squared Hellinger distance loss functions between measurement data and a given density matrix reconstruction, calculated using POVM expectations.
- Measurement operators for homodyne detection and displacement-parity measurement (Wigner function reconstruction).
- More formal simulation of finite measurements using `measure_shots()`, as an alternative to the existing additive Gaussian noise approach.
- Photon loss noise, mode mismatch noise, phase error noise, displacement error noise and dark count noise for heterodyne/Husimi Q measurement, along with improved demonstrations in the `Noise_demo.ipynb` example notebook.
- Support for negative cat states for both individual and batch states, controlled using the `positive` argument.
- The introduction of discrete variable (qubit) states to QSTToolkit for the first time: the N-qubit Hadamard state and density matrix, and functions for creating density matrices from QuTiP's Bell and GHZ states.
- `phase_space_grid` to define a NumPy meshgrid of phase space coordinates - prioritised over `xgrid` and `pgrid` in measurement operator functions (both options are supported).
- The `numpy` argument in `expectation`, allowing a choice between Tensor and array expectation probabilities and defaulting as `False`.
- Wigner function plots for density matrices, as well as Wigner plotting methods for all model classes.

### Changed

- All expectation values and measurement probabilities are normalized after creation, to ensure reconstruction using valid probability distributions (which sum to 1). This is done in model training loops, loss functions and data creation in example notebooks.
- Default parameters for `optical_state_dataset()`. `mixed_state_noise_level` now 0.1 and `amplification_ntherm` now 1.0.
- Measurement operators now contained in `.data.measurement`. Global and `.tomography` imports should not be affected
- Loss functions (including log likelihood) now contained in `.tomography.loss`. `.tomography.QST` import should not be affected.
- Improved error handling and docstring formatting.

### Deprecated

- The `dim` argument in all state creation except binomial states, in favor of `N`. All state batch creation keeps `dim`.
- The `density_matrix` argument throughout, in favor of `rho`.
- The `Husimi_Q` argument in `measurement_operators()`, in favor of `heterodyne`.
- The `dim_limit` argument for photon occupation measurement in `measurement_operators()`, in favor of `N_limit`.
- `gaussian_convolution`, in favor of `amplification_noise`.
- `variance`, in favor of `ntherm`.
- `Gaussian_conv_ntherm`, in favor of `amplification_ntherm`.
- `plot_Hinton()`, in favor of `plot_hinton()`, and all other capitalized spellings of Hinton in function names (e.g. `.plot_comparison_Hintons()`) in favor of the lower case spelling.
- `plot_Husimi_Q()`, in favor of `plot_husimi_Q()`, and all other capitalized spellings of Husimi in function names (e.g. `.plot_comparison_Husimi_Qs()`) in favor of the lower case spelling.

### Fixed

- Some deprecation warnings from SciPy and TensorFlow functions.

## v1.1.1 - 2025-04-14

### Added

- `optical_state_dataset()` now verbose during state initialization, as well as data generation.

### Changed

- More efficient synthesis of GKP states using `numpy.mgrid()`.
- Default mean value for additive Gaussian noise in `apply_measurement_noise()` is now 0.0, to better model error from finite measurements and discrete (binned) measurement pixels for Husimi Q data.
- Default standard deviation for additive Gaussian noise in `optical_state_dataset()` and example notebooks is now 0.01.
- Improved error handling in `qsttoolkit.quantum`, `qsttoolkit.tomography.QST` and `qsttoolkit.data.noise`.
- Updated example notebook cell outputs to reflect changes.
- Improvements to page intro text in documentation.

### Fixed

- Fixed an error in the GAN model's `build_generator()` function arising due to redundant function arguments for controlling the level of noise in the architecture.

## v1.1.0 - 2025-03-20

A general update providing better performing and more generalized and customizable implementations of the existing four models, each now running on common TensorFlow-based functions to ensure consistency and computational efficiency. This modular framework for modeling brings QSTToolkit closer to its goal of providing fully 'drag-and-drop' functions for tomography research. An expanded range of functions for states, noise and model performance analysis is provided, with additional improvements to plotting and documentation formatting.

### Added
- Functions (emulating those available in QuTiP) for directly initializing custom states as density matrices: `num_dm()`, `binomial_dm()`, `cat_dm()` and `gkp_dm()`, with the same arguments as their XXX_state counterparts.
- `salt_and_pepper_noise()` now allows for customisation of salt noise levels. The `prob` argument is replaced by `salt_p` and `pepper_p`.
- `optical_state_dataset()` now allows for customisation of state numbers and noise levels using the optional `state_numbers` and various noise level arguments. Gaussian convolution noise is now supported in the dataset.
- Verbosity control in CNN and Multitask model classes using the verbose argument (feeds values directly to `model.fit()`).
- Customisation of the GAN reconstruction using the new `gen_optimizer`, `disc_optimizer` and `loss_fn` arguments.
- Reconstruction time logging in the GAN and MLE models using the `time_log_interval` argument.
- `StateReconstructor.calculate_fidelities()` creates a new column `StateReconstructor.predictions_df['fidelity']`.
- `StateReconstructor.plot_fidelities()` plots a histogram of reconstruction fidelities.
- `.fidelity(true_dm)` method to calculate reconstruction fidelity for all QST model classes.
- `plot_Wigner()` for plotting the Wigner function of an optical quantum state.
- Axes, tick and label font size customisation in all plotting functions, plus colorbar presence now controllable where relevant.
- Regularisation and plot-formatting global utility functions.

### Changed
- QSTToolkit's SciPy-based MLE implementation has been replaced by a custom TensorFlow gradient descent-based training loop. The `MLEQuantumStateTomography` model is now formatted similarly to `GANQuantumStateTomography` for consistency and both the MLE and GAN implementations now use `reconstruct_density_matrix()`. This has led to a drastic improvement in the MLE's QST performance.
- The GAN generator network now outputs a lower triangular matrix, using a new custom `CholeskyLowerTriangular` TensorFlow network layer, to be interpreted as a Cholesky decomposition which may be inverted using `reconstruct_density_matrix()`.
- New default parameters in all models, giving improved overall performance.
- Core tomography functions (`parameterise_density_matrix()`, `reconstruct_density_matrix()`, `log_likelihood()`) are now built entirely using TensorFlow linear algebra submodules (`tf.linalg` and `tf.math`) in order to create a consistent TensorFlow-based `qsttoolkit.tomography` subpackage with computationally efficient and cross-compatible tomography models.
- Key attributes and methods (generally analysis and plotting) for tomography models are now established in a `QuantumStateTomography` parent class, and are inherited by each child model class. On occasion these standard methods can be overridden by the child class e.g. `GANQuantumStateTomography.plot_losses()` plots both generator and discriminator losses rather than a single loss function.
- `log_likelihood()` moved from `tomography.MLE` to `tomography.QST`. Globally imported function should not be affected.
- `expectation()` moved from `tomography.GAN_reconstructor.train` to the `qsttoolkit.quantum` submodule. Globally imported function should not be affected.
- Many functions (`mixed_state_noise()`, all plotting functions, `fidelity()`) are now less fussy about the input data type, and the input will be converted to the correct type internally.
- Some functions renamed to American English spellings: `States.normalise()` -> `States.normalize()`, `parameterise_density_matrix()` -> `parametrize_density_matrix()`. All old aliases are deprecated.
- `plot_Husimi_Q()` infers phase space extents from input grids.
- Dependency changes: numpy 1.26.4 -> 2.0.2, scipy 1.13.1 -> 1.14.1, scikit-learn 1.6.0 -> 1.6.1, tensorflow 2.17.1 -> 2.18.0.
- Extensive improvements to input error handling throughout.
- Improved consistency in docstring/documentation formatting.
- Updated example notebooks to reflect all changes and added model analysis functionality, as well as a new `Noise_demo.ipynb` example notebook demonstrating the application of QSTToolkit's range of noise sources.
- Minor changes to `README.md`.

### Deprecated
- The `N` argument in all state and state batch creation, in favor of `dim`.
- `.normalise()` in state batch creation classes, in favor of `.normalize()`.
- `salt_and_pepper_noise(prob)`, in favor of `salt_and_pepper_noise(salt_p, pepper_p)`.
- `apply_measurement_noise(salt_and_pepper_prob)`, in favor of `apply_measurement_noise(salt_p, pepper_p)`.
- The `dim` argument in all model classes (`Nc` in `StateReconstructor`) and core QST functions - system dimensionality is now inferred in situ when required.
- The `latent_dim` in all model_classes, in favor of `data_dim`.
- `MLEQuantumStateTomography.reconstruct(method)`, in favor of `MLEQuantumStateTomography.reconstruct(optimizer)` to identify a Keras-compatible optimizer for the reconstruction training.
- `MLEQuantumStateTomography.reconstruct(verbose)`, in favor of `MLEQuantumStateTomography.reconstruct(verbose_interval)`.
- `MLEQuantumStateTomography.plot_cost_values()`, in favor of `MLEQuantumStateTomography.plot_losses()`.
- `StateReconstructor.predictions_df['true_states']`, in favor of `StateReconstructor.predictions_df['true_dms']`.
- `parameterise_density_matrix()`, in favor of `parametrize_density_matrix()`.
- `measurement_operators(measurement_type='Husimi-Q')`, in favor of `measurement_operators(measurement_type='Husimi_Q')`.
- MLE constraint functions `trace_constraint()` and `positivity_constraint()` - no longer required for the TensorFlow-based MLE implementation.
- `plot_hinton()`, in favor of `plot_Hinton()`, and all other lowercase spellings of Hinton in function names (e.g. `.plot_comparison_hintons()`) in favor of the capitalized spelling.

### Removed
- The single tomography.MLE submodule - MLE now implemented in tomography.MLE_reconstructor.
- Loose framework for future batch tomography in the GAN training loop - may be added back in future.

### Fixed
- Various errors and edge cases in the `StateReconstructor` class methods, particularly in plotting functions.

## v1.0.0 - 2025-01-22

The initial release of QSTToolkit, providing custom optical quantum states, state batches and a pre-defined training dataset, noise sources, Convolutional Neural Network (CNN) quantum state discrimination, and quantum state tomography using Maximum Likelihood Estimation (MLE), Generative Adversarial Networks (GANs) and Multitask models.

### Added
- The `qsttoolkit.data` subpackage, providing synthetic data generation and noise simulation:
    - Num, Binomial, Cat and GKP states, building on the QuTiP framework for state preparation.
    - Object-oriented batch production of Fock, Coherent, Thermal, Num, Binomial, Cat, GKP and Random states of a given number, with parameters randomised within customizable limits.
    - State preparation noise (mixed state noise) applied to a density matrix.
    - Measurement and data noise (Gaussian convolution, affine transformations, additive Gaussian, and salt and pepper noise) applied to Q function image data.
    - A single `apply_measurement_noise()` function to apply all measurement and data noise sources at once in customizable levels.
    - A pre-defined dataset, created by `optical_state_dataset()`, designed as a standardized dataset for training machine learning models.
- The `qsttoolkit.tomography` subpackage, providing classes for one quantum state discrimination model and three quantum state reconstruction/tomography models:
    - Convolutional Neural Network (CNN) quantum state discrimination (built in TensorFlow).
    - Maximum Likelihood Estimation (MLE) quantum state tomography (built in TensorFlow).
    - Generative Adversarial Network (GAN) quantum state tomography (built using a custom TensorFlow training loop).
    - Multitask quantum state tomography (built in SciPy).
    - Performance analysis methods (e.g. `.plot_fidelities()`, `.plot_losses()`, `.evaluate_classification()`, `.plot_comparison_Hintons()`) for each model class.
    - Global QST utility functions:
        - Projective measurement operators for photon occupation number and displace-and-measure phase space Husimi Q measurements.
        - Cholesky decomposition parametrization and reconstruction.
        - Positive-semidefiniteness and unit-trace constraints.
- Additional submodules:
    - `qsttoolkit.quantum`: miscellaneous quantum physics tools: fidelity calculation, and two initial guesses for MLE (random PSD matrix and the maximally mixed state).
    - `qsttoolkit.plots`: preset formatting for plotting photon occupation numbers, density matrices (Hinton plots) and Husimi Q functions.