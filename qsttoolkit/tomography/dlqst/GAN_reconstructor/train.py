import tensorflow as tf

from qsttoolkit.quantum import fidelity


def expectation(rho, measurement_operators: list) -> tf.Tensor:
    """
    Computes the expectation values of the given density matrix with respect to the given projective measurement operators.
    Uses excusively TensorFlow data types and operations in order to integrate efficiently with the rest of the GAN training when using a GPU.

    Parameters
    ----------
    rho : tf.Tensor
        Density matrix to compute the expectation values for.
    measurement_operators : list of tf.Tensor
        List of projective measurement operators to compute the expectation values with respect to.

    Returns
    -------
    tf.Tensor
        Expectation values of the density matrix with respect to the measurement operators.
    """
    measurements = [tf.linalg.trace(tf.matmul(E, rho)) for E in measurement_operators]
    norm_real_measurements = tf.linalg.normalize(tf.math.real(measurements))[0]
    return tf.reshape(norm_real_measurements, (1, len(norm_real_measurements)))

def train(generator: tf.keras.Model, discriminator: tf.keras.Model, measurement_data: list, measurement_operators: list, epochs: int = 1000, verbose_interval: int = None, num_progress_saves: int = None, true_dm: tf.Tensor = None) -> tuple:
    """
    Trains the generator and discriminator networks adversarially using the given measurement data and projective measurement operators.

    Parameters
    ----------
    generator : tf.keras.Model
        The generator network.
    discriminator : tf.keras.Model
        The discriminator network.
    measurement_data : list of tf.Tensor
        The measurement data to train the networks on.
    measurement_operators : list of tf.Tensor
        The projective measurement operators corresponding to the measurement data.
    epochs : int
        The number of training epochs. Default is 1000.
    verbose_interval : int
        The interval at which to print progress updates. Default is None.
    num_progress_saves : int
        The number of intermediate progress saves to make. Default is None.

    Returns
    -------
    list of tf.Tensor
        The generator losses.
    list of tf.Tensor
        The discriminator losses.
    list of tf.Tensor
        The intermediate progress saves.
    list of tf.Tensor
        The fidelities of the generator density matrices with respect to the true density matrix.
    """
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    gen_losses = []
    disc_losses = []
    if num_progress_saves:
        progress_save_interval = epochs // num_progress_saves
        progress_saves = []
    else:
        progress_saves = None
    fidelities = [] if true_dm is not None else None

    for epoch in range(epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        if true_dm is not None: epoch_fidelity = 0.0

        steps = len(measurement_data)

        for i in range(steps):
            # Select a single real vector
            real_measurements = measurement_data[i]

            # Forward pass through generator
            # noise = tf.random.normal([1, real_v.shape[1]])  # Shape (1, latent_dim)
            with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
                # Generator output - density matrix
                generated_dm = generator(real_measurements)

                # Calculate expectations
                generated_measurements = expectation(generated_dm, measurement_operators)

                # Discriminator outputs
                reconstructed_preds = discriminator(generated_measurements)  # Reconstructed data vector probability
                real_preds = discriminator(real_measurements)  # Original data vector probability

                # Loss functions
                disc_loss = (loss_fn(tf.ones_like(real_preds), real_preds) + loss_fn(tf.zeros_like(reconstructed_preds), reconstructed_preds)) / 2
                gen_loss = loss_fn(tf.ones_like(reconstructed_preds), reconstructed_preds)

                # Fidelities
                if true_dm is not None: step_fidelity = fidelity(generator(real_measurements), true_dm)

            # Backpropagation
            grads_disc = tape_disc.gradient(disc_loss, discriminator.trainable_weights)
            disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

            grads_gen = tape_gen.gradient(gen_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_weights))

            # Accumulate losses
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss

            # Accumulate fidelities
            if true_dm is not None: epoch_fidelity += step_fidelity

        # Calculate average losses for the epoch
        avg_gen_loss = epoch_gen_loss / steps
        avg_disc_loss = epoch_disc_loss / steps

        # Append losses to the lists
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        # ...for fidelities
        if true_dm is not None:
            avg_fidelity = epoch_fidelity
            fidelities.append(avg_fidelity)

        # Save progress
        if num_progress_saves and epoch % progress_save_interval == 0:
            progress_saves.append(generator(real_measurements))

        # Log progress
        if verbose_interval and epoch % verbose_interval == 0:
            print(f"Epoch {epoch}, Generator Loss: {avg_gen_loss.numpy()}, Discriminator Loss: {avg_disc_loss.numpy()}")

    return gen_losses, disc_losses, progress_saves, fidelities