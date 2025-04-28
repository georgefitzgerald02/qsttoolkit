import tensorflow as tf

from qsttoolkit.tomography.QST import reconstruct_density_matrix, log_likelihood
from qsttoolkit.quantum import fidelity, expectation


def train(model: tf.keras.Model, measurement_data: list, measurement_operators: list, epochs: int = 100, optimiser = None, L1_reg=0.0, verbose_interval: int = None, num_progress_saves: int = None, true_dm: tf.Tensor = None) -> tuple:
    """
    Trains the model and discriminator networks adversarially using the given measurement data and projective measurement operators.

    Parameters
    ----------
    model : tf.keras.Model
        The encoding network.
    measurement_data : list of tf.Tensor
        The measurement data to train the network on.
    measurement_operators : list of Qobj
        The projective measurement operators corresponding to the measurement data.
    epochs : int
        The number of training epochs. Defaults to 100.
    optimiser : tf.keras.optimizers.Optimiser
        The optimiser to use for the model. Defaults to Adam with learning rate 0.0002.
    L1_reg : float
        L1 regularisation parameter. Defaults to 0.
    verbose_interval : int
        The interval at which to print progress updates. Defaults to None.
    num_progress_saves : int
        The number of intermediate progress saves to make. Defaults to None.
    true_dm : tf.Tensor
        The true density matrix used for calculating fidelities. Defaults to None

    Returns
    -------
    list of tf.Tensor
        The training losses.
    list of tf.Tensor
        The intermediate progress saves.
    list of tf.Tensor
        The fidelities of the reconstructed density matrices with respect to the true density matrix.
    """
    optimiser = optimiser if optimiser else tf.keras.optimizers.Adam(learning_rate=0.0002)
    
    losses = []
    if num_progress_saves:
        progress_save_interval = epochs // num_progress_saves
        progress_saves = []
    else:
        progress_saves = None
    fidelities = [] if true_dm is not None else None

    for epoch in range(epochs):
        epoch_loss = 0.0
        if true_dm is not None: epoch_fidelity = 0.0

        steps = len(measurement_data)

        for i in range(steps):
            # Select a single vector
            real_measurements = measurement_data[i]

            # Forward pass through model
            # noise = tf.random.normal([1, real_v.shape[1]])  # Shape (1, latent_dim)
            with tf.GradientTape() as tape:
                # Model output - Cholesky decomposition
                parameterised_generated_dm = model(real_measurements)

                # Invert Cholesky decomposition
                generated_dm = reconstruct_density_matrix(parameterised_generated_dm)

                # Loss function
                loss = log_likelihood(generated_dm, real_measurements, measurement_operators, L1_reg=L1_reg)

                # Fidelities
                if true_dm is not None: step_fidelity = fidelity(generated_dm[i], true_dm)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_weights)
            optimiser.apply_gradients(zip(grads, model.trainable_weights))

            # Accumulate losses
            epoch_loss += loss

            # Accumulate fidelities
            if true_dm is not None: epoch_fidelity += step_fidelity

        # Calculate average losses for the epoch
        avg_loss = epoch_loss / steps

        # Append losses to the lists
        losses.append(avg_loss)

        # ...for fidelities
        if true_dm is not None:
            avg_fidelity = epoch_fidelity
            fidelities.append(avg_fidelity)

        # Save progress
        if num_progress_saves and epoch % progress_save_interval == 0:
            progress_saves.append(reconstruct_density_matrix(model(real_measurements)).numpy())

        # Log progress
        if verbose_interval and epoch % verbose_interval == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss.numpy()}, Fidelity: {avg_fidelity}")

    return losses, progress_saves, fidelities