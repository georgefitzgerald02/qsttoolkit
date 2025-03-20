import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import warnings

from qsttoolkit.tomography.dlqst.CNN_classifier.architecture import build_classifier


class CNNQuantumStateDiscrimination:
    """
    A class for training and evaluating a CNN classifier for quantum state discrimination.

    Attributes
    ----------
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder.
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped. Defaults to 5.
    lr_scheduler_factor : float
        Factor by which the learning rate will be reduced. Defaults to 0.5.
    lr_scheduler_patience : int
        Number of epochs with no improvement after which learning rate will be reduced. Defaults to 3.
    """
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, label_encoder: sklearn.preprocessing.LabelEncoder, early_stopping_patience: int=5, lr_scheduler_factor: float=0.5, lr_scheduler_patience: int=3, dim=None):
        if dim: warnings.warn("dim is no longer required for this class and will be removed in a future version.", DeprecationWarning, stacklevel=2)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.data_dim = X_train[0].shape[0]

        self.model = build_classifier(data_input_shape=(self.data_dim, self.data_dim, 1))
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_scheduler_factor, patience=lr_scheduler_patience)
        self.callbacks = [self.early_stopping, self.lr_scheduler]

    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics: list[str]=['accuracy'], epochs: int=20, batch_size: int=32, validation_split: float=0.2, verbose='auto'):
        """
        Compiles and trains the model.

        Parameters
        ----------
        optimizer
            Optimizer to use in the training. Defaults to 'adam'.
        loss
            Loss function to use in the training. Defaults to 'sparse_categorical_crossentropy'.
        metrics : list[str]
            Metrics to measure model performance during training. Defaults to ['accuracy'].
        epochs : int
            Number of epochs to train the model. Defaults to 20.
        batch_size : int
            Training batch size. Defaults to 32.
        validation_split : float
            Fraction of the training data to use as validation data. Defaults to 0.2.
        verbose
            Verbosity mode. Defaults to 'auto'.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

        self.history = self.model.fit(self.X_train, self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      callbacks=self.callbacks,
                                      verbose=verbose)

    def plot_training(self):
        """
        Plots the training and validation accuracy and loss.
        """
        _, axs = plt.subplots(1, len(self.metrics)+1, figsize=(6*(len(self.metrics)+1), 5))

        # Plot training & validation accuracy
        for i, metric in enumerate(self.metrics):
            axs[i].plot(self.history.history['accuracy'], label='train accuracy')
            axs[i].plot(self.history.history['val_accuracy'], label='val accuracy')
            if metric == 'accuracy': axs[i].set_ylim(0,1)
            axs[i].set_title('Model Metrics')
            axs[i].set_ylabel(self.metrics[0].capitalize())
            axs[i].set_xlabel('Epoch')
            axs[i].legend()

        # Plot training & validation loss
        axs[-1].plot(self.history.history['loss'], label='train loss')
        axs[-1].plot(self.history.history['val_loss'], label='validation loss')
        if self.loss == 'sparse_categorical_crossentropy': axs[-1].set_ylim(0,1)
        axs[-1].set_title('Model Loss')
        axs[-1].set_ylabel('Loss')
        axs[-1].set_xlabel('Epoch')
        axs[-1].legend()

        plt.show()
    
    def evaluate_classification(self, include_confusion_matrix: bool=True, include_classification_report: bool=True):
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
            # Confusion matrix and plot
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
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
            Quantum states to classify.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        X = np.array([x for x in X]).reshape(-1, self.data_dim, self.data_dim, 1)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)