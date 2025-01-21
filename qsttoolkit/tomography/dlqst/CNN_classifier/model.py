import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from qsttoolkit.tomography.dlqst.CNN_classifier.architecture import build_classifier


class CNNQuantumStateDiscrimination:
    """
    A class for training and evaluating a CNN classifier for quantum state discrimination.

    Attributes
    ----------
    dim : int
        The Hilbert space dimensionality.
    X_train : np.ndarray
        The training data.
    X_test : np.ndarray
        The test data.
    y_train : np.ndarray
        The training labels.
    y_test : np.ndarray
        The test labels.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder.
    early_stopping_patience : int
        The number of epochs with no improvement after which training will be stopped.
    lr_scheduler_factor : float
        Factor by which the learning rate will be reduced.
    lr_scheduler_patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    """
    def __init__(self, dim: int, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, label_encoder, early_stopping_patience=5, lr_scheduler_factor=0.5, lr_scheduler_patience=3):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.dim = dim
        self.latent_dim = X_train[0].shape[0]

        self.model = build_classifier(data_input_shape=(self.latent_dim, self.latent_dim, 1))
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_scheduler_factor, patience=lr_scheduler_patience)
        self.callbacks = [self.early_stopping, self.lr_scheduler]

    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=20, batch_size=32, validation_split=0.2):
        """
        Compiles and trains the model.

        Parameters
        ----------
        optimizer : str
            The optimizer to use in the training. Defaults to 'adam'.
        loss : str
            The loss function to use in the training. Defaults to 'sparse_categorical_crossentropy'.
        metrics : list
            The metrics to measure model performance during training. Defaults to ['accuracy'].
        epochs : int
            The number of epochs to train the model. Defaults to 20.
        batch_size : int
            The training batch size. Defaults to 32.
        validation_split : float
            The fraction of the training data to use as validation data. Defaults to 0.2.
        """
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

        self.history = self.model.fit(self.X_train, self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      callbacks=self.callbacks)

    def plot_training(self):
        """
        Plots the training and validation accuracy and loss.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot training & validation accuracy
        axs[0].plot(self.history.history['accuracy'], label='train accuracy')
        axs[0].plot(self.history.history['val_accuracy'], label='val accuracy')
        axs[0].set_ylim(0,1)
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend()

        # Plot training & validation loss
        axs[1].plot(self.history.history['loss'], label='train loss')
        axs[1].plot(self.history.history['val_loss'], label='val loss')
        axs[1].set_ylim(0,1)
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend()

        plt.show()
    
    def evaluate_classification(self, include_confusion_matrix=True, include_classification_report=True):
        """
        Evaluates the model on the test data.

        Parameters
        ----------
        include_confusion_matrix : bool
            Whether to include the confusion matrix in the evaluation. Defaults to True.
        include_classification_report : bool
            Whether to include the classification report in the evaluation. Defaults to True.
        """
        y_pred = np.argmax(self.model.predict(self.X_test), axis=1)

        if include_confusion_matrix:
            # Extract labels
            y_labels = self.label_encoder.inverse_transform(self.y_test)

            # Confusion matrix and plot
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        if include_classification_report:
            # Classification report
            class_report = classification_report(self.y_test, y_pred)
            print("Classification Report:")
            print(class_report)

    def classify(self, X: np.ndarray) -> np.ndarray:
        """
        Classifies a set of quantum states using the trained model.

        Parameters
        ----------
        X : np.ndarray
            The quantum states to classify.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        X = np.array([x for x in X]).reshape(-1, self.latent_dim, self.latent_dim, 1)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)