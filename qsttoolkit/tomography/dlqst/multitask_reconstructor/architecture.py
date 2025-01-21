import tensorflow as tf
from tensorflow.keras import layers, Model


def build_feature_extractor(input_shape: tuple) -> tf.keras.Model:
    """
    Builds the feature extractor branch of the multitask reconstructor model.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.

    Returns
    -------
    tf.keras.Model
        The feature extractor model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.GaussianNoise(0.1)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(128, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.GaussianNoise(0.1)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(256, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.Conv2D(512, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.2)(x)

    feature_vector = layers.Flatten()(x)
    return Model(inputs, feature_vector, name="feature_extractor")

def build_classification_tail(input_features: tf.Tensor, num_classes: int) -> tf.Tensor:
    """
    Builds the classification tail of the multitask reconstructor model.

    Parameters
    ----------
    input_features : tf.Tensor
        The input feature vector.
    num_classes : int
        The number of classes to classify.
    
    Returns
    -------
    tf.Tensor
        The output tensor of the classification tail.
    """
    x = layers.Dense(64)(input_features)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)
    return output

# Regression tail                   Future work could implement two networks, one to predict each of the real and complex parts of the parameter
def build_regression_tail(input_features: tf.Tensor, num_outputs: int) -> tf.Tensor:
    """
    Builds the regression tail of the multitask reconstructor model.

    Parameters
    ----------
    input_features : tf.Tensor
        The input feature vector.
    num_outputs : int
        The number of regression outputs.

    Returns
    -------
    tf.Tensor
        The output tensor of the regression tail.
    """
    x = layers.Dense(64)(input_features)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    output = layers.Dense(num_outputs, name="regression_output")(x)
    return output

def build_multitask_reconstructor(input_shape: tuple, num_classes: int, num_regression_outputs: int) -> tf.keras.Model:
    """
    Builds the multitask reconstructor model.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    num_classes : int
        The number of classes to classify.
    num_regression_outputs : int
        The number of regression outputs.

    Returns
    -------
    tf.keras.Model
        The multitask reconstructor model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    feature_extractor = build_feature_extractor(input_shape)
    feature_vector = feature_extractor(inputs)

    classification_tail = build_classification_tail(feature_vector, num_classes)
    regression_input = layers.Concatenate()([feature_vector, classification_tail])
    regression_tail = build_regression_tail(regression_input, num_regression_outputs)

    return Model(inputs=inputs, outputs=[classification_tail, regression_tail])