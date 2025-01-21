import tensorflow as tf
from tensorflow.keras import layers, Model


def build_classifier(data_input_shape: tuple) -> tf.keras.Model:
    """
    Builds a simple CNN classifier for QST data.
    
    Parameters
    ----------
    data_input_shape : tuple
        The shape of the input data.
        
    Returns
    -------
    tf.keras.Model
        The classifier model.
    """
    input = layers.Input(shape=data_input_shape)

    # Convolutional and Pooling layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(7, activation='softmax')(x)

    return Model(input, x)