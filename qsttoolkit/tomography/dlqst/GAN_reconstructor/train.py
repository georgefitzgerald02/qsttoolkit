import tensorflow as tf
import time

from qsttoolkit.tomography.QST import reconstruct_density_matrix
from qsttoolkit.quantum import fidelity, expectation


def train(generator: tf.keras.Model, discriminator: tf.keras.Model, measurement_data: list, measurement_operators: list, epochs: int=100, gen_optimizer: tf.keras.optimizers.Optimizer=None, disc_optimizer: tf.keras.optimizers.Optimizer=None, loss_fn: tf.keras.losses.Loss=tf.keras.losses.BinaryCrossentropy(), verbose_interval: int=None, num_progress_saves: int=None, true_dm: tf.Tensor=None, time_log_interval: int=None) -> tuple:
    """
    Trains the generator and discriminator networks adversarially using the given measurement data and projective measurement operators.

    Parameters
    ----------
    generator : tf.keras.Model
        Generator network.
    discriminator : tf.keras.Model
        Discriminator network.
    measurement_data : list of np.ndarray
        Frequency of each measurement outcome.
    measurement_operators : list of Qobj
        Projective operators corresponding to the measurement outcomes.
    epochs : int
        Number of epochs to train for. Defaults to 100.
    gen_optimizer : tf.keras.optimizers.Optimizer
        Generator optimizer. Defaults to Adam with learning rate 0.0002.
    disc_optimizer : tf.keras.optimizers.Optimizer
        Discriminator optimizer. Defaults to Adam with learning rate 0.0002.
    loss_fn : tf.keras.losses.Loss
        Loss function to use. Defaults to BinaryCrossentropy.
    verbose_interval : int
        Interval at which to print progress updates. Defaults to None.
    num_progress_saves : int
        Number of intermediate progress saves to make. Defaults to None.
    true_dm : tf.Tensor
        True density matrix used for calculating fidelities. Defaults to None.
    time_log_interval : int
        Interval at which to log the time taken after each epoch. Defaults to None.

    Returns
    -------
    list of tf.Tensor
        Generator losses.
    list of tf.Tensor
        Discriminator losses.
    list of np.ndarray
        Intermediate progress saves.
    list of float
        Fidelities of the generator density matrices with respect to the true density matrix.
    list of float
        Time taken after each epoch.
    """
    gen_optimizer = gen_optimizer if gen_optimizer else tf.keras.optimizers.Adam(learning_rate=0.0002)
    disc_optimizer = disc_optimizer if disc_optimizer else tf.keras.optimizers.Adam(learning_rate=0.0002)

    gen_losses = []
    disc_losses = []
    if num_progress_saves:
        progress_save_interval = epochs // num_progress_saves
        progress_saves = []
    else:
        progress_saves = None
    fidelities = [] if true_dm is not None else None
    if time_log_interval:
        start_time = time.time()
        times = []
    else:
        times = None

    for epoch in range(epochs):
        # Forward pass through generator
        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
            # Generator output - Cholesky decomposition
            parametrized_generated_dm = generator(measurement_data)

            # Invert Cholesky decomposition
            generated_dm = reconstruct_density_matrix(parametrized_generated_dm)

            # Expectation values
            generated_measurements = expectation(generated_dm, measurement_operators)
            generated_measurements /= tf.reduce_sum(generated_measurements)  # Normalize to 1

            # Discriminator outputs
            reconstructed_preds = discriminator(generated_measurements)  # Reconstructed data vector probability
            real_preds = discriminator(measurement_data)  # Original data vector probability

            # Loss functions
            epoch_disc_loss = (loss_fn(tf.ones_like(real_preds), real_preds) + loss_fn(tf.zeros_like(reconstructed_preds), reconstructed_preds)) / 2
            epoch_gen_loss = loss_fn(tf.ones_like(reconstructed_preds), reconstructed_preds)

            # Fidelities
            if true_dm is not None: epoch_fidelity = fidelity(generated_dm[0].numpy(), true_dm)

        # Backpropagation
        grads_disc = tape_disc.gradient(epoch_disc_loss, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        grads_gen = tape_gen.gradient(epoch_gen_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_weights))

        # Append losses to the lists
        gen_losses.append(epoch_gen_loss)
        disc_losses.append(epoch_disc_loss)

        # ...for fidelities
        if true_dm is not None:
            fidelities.append(epoch_fidelity)

        # Save progress
        if num_progress_saves and epoch % progress_save_interval == 0:
            progress_saves.append(generated_dm[0].numpy())

        # Log progress
        if verbose_interval and epoch % verbose_interval == 0:
            print(f"Epoch {epoch}/{epochs}, Generator Loss: {epoch_gen_loss.numpy()}, Discriminator Loss: {epoch_disc_loss.numpy()}, Fidelity: {epoch_fidelity if true_dm is not None else None}")

        # Log time
        if time_log_interval and epoch % time_log_interval == 0:
            times.append(time.time() - start_time)

    return gen_losses, disc_losses, progress_saves, fidelities, times