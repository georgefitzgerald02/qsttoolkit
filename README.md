# QSTToolkit

QSTToolkit is an open-source package for performing optical quantum state tomography (QST) research using both traditional (statistical) and novel deep learning-powered methods in Python. Key functionality includes:
- Fast, compute-efficient and customisable generation of realistic synthetic data for a variety of optical quantum states using the [QuTiP](https://qutip.org/docs/4.0.2/index.html) package.
- Maximum Likelihood Estimation quantum state tomography.
- A variety of deep learning powered methods for quantum state discrimination and tomography.

The key aim of QSTToolkit is to create a standard framework for researching, designing and comparing methods for quantum state tomography in noisy, high-dimensional, continuous quantum systems. Recently proposed models are implemeted, using standardised synthetic data to allow for fully valid comparison between approaches.

This work is the culmination of a physics masters project by George FitzGerald (gwfitzg@hotmail.com) at [Durham University's Department of Physics](https://www.durham.ac.uk/departments/academic/physics/).

## Table of Contents
- [Setup](#setup)
    - [Local Installation](#local-installation)
    - [Google Colab](#google-colab)
- [Usage](#usage)
    - [Importing QSTToolkit](#importing-qsttoolkit)
    - [Synthetic Data Generation](#synthetic-data-generation)
    - [Quantum State Tomography](#quantum-state-tomography)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [Documentation](#documentation)
- [Future Development](#future-development)
- [License](#license)
- [Contributing](#contributing)
- [References](#references)

## Setup

QSTToolkit is available via the [PyPi](https://pypi.org/) package manager.

### Local Installation

 **To install QSTToolkit** and run the example notebooks in a local environment (not recommended for the deep learning models without GPU access):

1. Clone the package repository (only needed if you plan to run the example notebooks or inspect the source code):
    ```bash
    git clone https://github.com/georgefitzgerald02/qsttoolkit.git
    cd qsttoolkit
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install QSTToolkit:
    ```bash
    pip install qsttoolkit
    ```

4. If you plan to run the example notebooks, create an iPython kernel in your virtual environment which allows the installed packages to be accessed by the notebooks:
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=qsttoolkit_kernel
    ```
    `qsttoolkit_kernel` should now be available to select from the list of available kernels in the Jupyter notebook interface.

### Google Colab

**To use QSTToolkit in [Google Colab](https://colab.research.google.com/)**, run the following cell once each time you open your project in order to install `qsttoolkit` to the runtime:

```python
!pip install qsttoolkit
```

**To run the example notebooks in Google Colab**, click *File*, *Open notebook*, *GitHub*, and in the *Enter a GitHub URL or search by organisation or user* box, paste the URL of this repository: (https://github.com/georgefitzgerald02/qsttoolkit). Then navigate to the example notebook of your choice and open it. The above cell is included at the start of each example notebook and should be run once upon opening.

## Usage

### Importing QSTToolkit

The features of QSTToolkit are organised into two main subpackages, `qsttoolkit.data` and `qsttoolkit.tomography`, along with additional miscellaneous modules such as `qsttoolkit.plots` and `qsttoolkit.quantum`. In the example notebooks, features that belong to one of the main subpackages are called from their subpackage, to demonstrate their location in the overall package. Some example lines of code written using QSTToolkit:

```python
import qsttoolkit as qst
```
```python
cat_batch = qst.data.CatStates(n_states = 1000, N = dim, alpha_magnitude_range = [0, 10])
```
```python
MLE_reconstructor = qst.tomography.MLEQuantumStateTomography(dim)
```
```python
print(qst.fidelity(test_state.full(), MLE_reconstructor.reconstructed_dm))
```

However, **all public classes and functions in QSTToolkit can also be called directly** from `qsttoolkit`, for example:

```python
import qsttoolkit as qst

cat_batch = qst.CatStates(n_states = 1000, N = dim, alpha_magnitude_range = [0, 10])
```

### Synthetic Data Generation (`qsttoolkit.data`)

QSTToolkit provides an expansion to the existing [QuTiP](https://qutip.org/docs/4.0.2/index.html) framework for producing synthetic state vectors and density matrices (generally in the form of [NumPy arrays](https://numpy.org/doc/2.1/reference/generated/numpy.array.html)) for optical quantum states, with a specific focus on producing realistic data suitable for training deep learning quantum state discrimination and tomography models. On top of [Fock](https://en.wikipedia.org/wiki/Fock_state), [coherent](https://en.wikipedia.org/wiki/Coherent_state), thermal and random states which can be produced directly using [QuTiP functions](https://qutip.org/docs/4.0.2/apidoc/functions.html), QSTToolkit provides functions for synthesizing specific useful superpositions of Fock and coherent states. The custom states currently provided are:
- Num states [[1]](#references): `data.states.num_state()`
- Binomial states [[1]](#references): `data.states.binomial_state()`
- [Cat states](https://en.wikipedia.org/wiki/Cat_state): `data.states.cat_state()`
- Gottesman-Kitaev-Preskill (GKP) states [[2]](#references): `data.states.gkp_state()`

These states can be produced individually, in batches of specified size with randomised state parameters, or in specific preset datasets, intended to be standard datasets for modelling. States can be produced as pure states, or with some mixing applied to the density matrix. Measurement data is generated for direct photon occupation number measurement, or a Husimi-Q function as a result of a displace-and-measure technique [[3]](#references). Different sources of noise can be applied to the image data for the latter at customisable levels.

Support for more states, measurement regimes and noise sources are planned for development. To request any specific features that might be useful for your work, please contact George FitzGerald at (gwfitzg@hotmail.com).

### Quantum State Tomography (`qsttoolkit.tomography`)

QSTToolkit currently provides classes to compile and train/optimise four models:

- [Convolutional neural network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) powered quantum state discrimination [[3]](#references): `tomography.dlqst.CNNQuantumStateDiscrimination`
- [Maximum likelihood estimation (MLE)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) based quantum state tomography: `tomography.tradqst.MLEQuantumStateTomography`
- [Generative adversarial network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) quantum state tomography [[4]](#references): `tomography.dlqst.GANQuantumStateTomography`
- [Multitasking](https://en.wikipedia.org/wiki/Multi-task_learning) classification/regression network quantum state characterisation [[5]](#references): `tomography.dlqst.MultitaskQuantumStateTomography`

The usage of each class varies depending on the model's composition and functionality. The `/example_notebooks` directory contains example Jupyter notebooks which run through the usage of each model, with example synthetic data preparation for the model's specific use case.

## Dependencies

- numpy 1.26.4
- scipy 1.13.1
- pandas 2.2.2
- matplotlib 3.10.0
- seaborn 0.13.2
- qutip 5.1.1
- scikit-learn 1.6.0
- tensorflow 2.17.1

## Directory Structure

```
qsttoolkit/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   ├── noise.py
│   ├── num_state_coeffs.py
│   ├── state_batches.py
│   └── states.py
├── tomography/
│   ├── __init__.py
│   ├── tradqst/
│   │   ├── __init__.py
│   │   └── MLE.py
│   ├── dlqst/
│   │   ├── __init__.py
│   │   ├── CNN_classifier/
│   │   │   ├── __init__.py
│   │   │   ├── architecture.py
│   │   │   └── model.py
│   │   ├── GAN_reconstructor/
│   │   │   ├── __init__.py
│   │   │   ├── architecture.py
│   │   │   ├── model.py
│   │   │   └── train.py
│   │   └── multitask_reconstructor/
│   │       ├── __init__.py
│   │       ├── architecture.py
│   │       ├── model.py
│   │       └── reconstruction.py
│   └── QST.py
├── plots.py
├── quantum.py
└── utils.py
```

## Documentation

Comprehensive docstrings can be found throughout the source code. Online documentation coming soon.

## Future Development

Planned new features coming soon:
- More traditional QST methods (Linear Inversion, Bayesian Inference, Compressed Sensing)
- More deep learning quantum state analyses (autoencoder, restricted Boltzmann machine (RBM))
- Generalisation to qubit tomography

## License

This project is licensed under the MIT License. You are free to:

- Use this code for personal and commercial purposes.
- Modify and distribute the code, as long as you include the original copyright notice and license text.

For more details, see the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests, improvements, or bug fixes.

To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Open a pull request.

For any other questions, please contact me at gwfitzg@hotmail.com

## References

References to key papers that inspired this package:

1. M. H. Michael et al., *New Class of Quantum Error-Correcting Codes for a Bosonic Mode* (2016), https://doi.org/10.1103/PhysRevX.6.031006
2. D. Gottesman, A. Kitaev and J. Preskill, *Encoding a qubit in an oscillator* (2001), https://doi.org/10.1103/PhysRevA.64.012310
3. S. Ahmed, C. Sánchez Muñoz, F. Nori, and A. F. Kockum, *Classification and reconstruction of optical quantum states with deep neural networks* (2021), https://doi.org/10.1103/PhysRevResearch.3.033278
4. S. Ahmed, C. S. Muñoz, F. Nori, and A. F. Kockum, *Quantum State Tomography with Conditional Generative Adversarial Networks* (2021), https://doi.org/10.1103/PhysRevLett.127.140502
5.  N. T. Luu, T. C. Truong, and D. T. Luu, *Universal quantum tomography with deep neural networks* (2024), https://doi.org/10.48550/arXiv.2407.01734