import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import warnings

from qsttoolkit.tomography.dlqst.multitask_reconstructor.architecture import build_multitask_reconstructor


class MultitaskQuantumStateTomography:
    """
    A class for training and evaluating a multitask regression/classification model for quantum state tomography.

    Attributes
    ----------
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test data.
    y_train : dict
        Dictionary containing both the training classification and regression labels.
    y_test : dict
        Dictionary containing both the test classification and regression labels.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder for the classification labels.
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped. Defaults to 30.
    lr_scheduler_factor : float
        Factor by which the learning rate will be reduced. Must be < 1. Defaults to 0.5.
    lr_scheduler_patience : int
        Number of epochs with no improvement after which learning rate will be reduced. Defaults to 15.
    """
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, label_encoder: sklearn.preprocessing.LabelEncoder, early_stopping_patience: int=30, lr_scheduler_factor: float=0.5, lr_scheduler_patience: int=15, dim=None):
        if dim: warnings.warn("dim is no longer required for this class and will be removed in a future version.", DeprecationWarning, stacklevel=2)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.data_dim = X_train[0].shape[0]

        self.model = build_multitask_reconstructor(input_shape=(self.data_dim, self.data_dim, 1), num_classes=len(self.label_encoder.classes_), num_regression_outputs=2)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_scheduler_factor, patience=lr_scheduler_patience)
        self.callbacks = [self.early_stopping, self.lr_scheduler]

    def train(self, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999), classification_loss='sparse_categorical_crossentropy', regression_loss='mse', classification_loss_weight: float=1.0, regression_loss_weight: float=0.5, classification_metric: str='accuracy', regression_metric: str='mae', epochs: int=30, batch_size: int=32, validation_split: float=0.2, verbose='auto'):
        """
        Compiles and trains the model.

        Parameters
        ----------
        optimizer
            Optimizer to use in the training. Defaults to Adam with learning rate 0.0005, beta_1=0.9, and beta_2=0.999.
        classification_loss
            Classification loss function to use in the training. Defaults to 'sparse_categorical_crossentropy'.
        regression_loss
            Regression loss function to use in the training. Defaults to 'mse'.
        classification_loss_weight : float
            Weight of the classification loss in the total loss. Defaults to 1.0.
        regression_loss_weight : float
            Weight of the regression loss in the total loss. Defaults to 1.0.
        classification_metric : str
            Metric to measure classification performance during training. Defaults to 'accuracy'.
        regression_metric : str
            Metric to measure regression performance during training. Defaults to 'mse'.
        epochs : int
            Number of epochs to train the model. Defaults to 30.
        batch_size : int
            Training batch size. Defaults to 32.
        validation_split : float
            Fraction of the training data to use as validation data. Defaults to 0.2.
        verbose
            Verbosity mode. Defaults to 'auto'.
        """
        self.optimizer = optimizer
        self.classification_loss = classification_loss
        self.regression_loss = regression_loss
        self.classification_loss_weight = classification_loss_weight
        self.regression_loss_weight = regression_loss_weight
        self.classification_metric = classification_metric  
        self.regression_metric = regression_metric

        self.model.compile(optimizer=optimizer,
                            loss={
                                "classification_output": classification_loss,
                                "regression_output": regression_loss
                            },
                            loss_weights={
                                "classification_output": classification_loss_weight,
                                "regression_output": regression_loss_weight
                            },
                            metrics={
                                "classification_output": classification_metric,
                                "regression_output": regression_metric
                            })

        self.history = self.model.fit(self.X_train, self.y_train,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      callbacks=self.callbacks,
                                      verbose=verbose)

    def plot_training(self):
        """Plots the training history over epochs."""
        _, axs = plt.subplots(1, 3, figsize=(20, 5))

        # Plot training & validation classsification metrics
        axs[0].plot(self.history.history[f"classification_output_{self.classification_metric}"], label=f"train {self.classification_metric}")
        axs[0].plot(self.history.history[f"val_classification_output_{self.classification_metric}"], label=f"validation {self.classification_metric}")
        if self.classification_metric == 'accuracy': axs[0].set_ylim(0,1)
        axs[0].set_title(f"Model {self.classification_metric.capitalize()}")
        axs[0].set_ylabel(self.classification_metric.capitalize())
        axs[0].set_xlabel('Epoch')
        axs[0].legend()

        # Plot training & validation regression metrics
        axs[1].plot(self.history.history[f"regression_output_{self.regression_metric}"], label=f"train {self.regression_metric}")
        axs[1].plot(self.history.history[f"val_regression_output_{self.regression_metric}"], label=f"validation {self.regression_metric}")
        axs[1].set_title(f"Model {self.regression_metric}")
        axs[1].set_ylabel(self.regression_metric)
        axs[1].set_xlabel('Epoch')
        axs[1].legend()

        # Plot training & validation composite loss
        axs[2].plot(self.history.history['loss'], label='train loss')
        axs[2].plot(self.history.history['val_loss'], label='validation loss')
        axs[2].set_title('Model Composite Loss')
        axs[2].set_ylabel('Composite Loss')
        axs[2].set_xlabel('Epoch')
        axs[2].legend()

        plt.show()

    def evaluate_classification(self, include_confusion_matrix: bool=True, include_classification_report: bool=True):
        """
        Evaluates the classification performance of the model.

        Parameters
        ----------
        include_confusion_matrix : bool
            Whether to include the confusion matrix in the evaluation. Defaults to True.
        include_classification_report : bool
            Whether to include the classification report in the evaluation. Defaults to True.
        """
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions[0], axis=1)

        if include_confusion_matrix:
            # Confusion matrix and plot
            cm = confusion_matrix(self.y_test['classification_output'], y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.show()

        if include_classification_report:
            # Classification report
            class_report = classification_report(self.y_test['classification_output'], y_pred)
            print("Classification Report:")
            print(class_report)

    def evaluate_regression(self):
        """Evaluates the regression performance of the model by plotting the predictions vs true values for each class."""
        predictions = self.model.predict(self.X_test)
        fig, axs = plt.subplots(1, len(self.label_encoder.classes_), figsize=(25, 4))
        fig.suptitle("Regression predictions vs true values for each true class")
        axs[0].set_ylabel("Predicted value")
        for i, ax in enumerate(axs.flat):
            true_values = self.y_test['regression_output'][self.y_test['classification_output'] == i]
            pred_values = predictions[1][self.y_test['classification_output'] == i]
            ax.scatter(true_values, pred_values)
            maximum = max(np.max(true_values), np.max(pred_values))
            minimum = min(np.min(true_values), np.min(pred_values))
            ax.plot([minimum, maximum], [minimum, maximum], color='red', linestyle='--')
            ax.set_title(self.label_encoder.classes_[i])
            ax.set_xlabel("True value")
        plt.show()

    def infer(self, data: np.ndarray):
        """
        Infers the quantum state label and key parameter from input data.

        Parameters
        ----------
        data : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predicted quantum state labels.
        """
        predictions = self.model.predict(data)
        y_pred = np.argmax(predictions[0], axis=1)
        predicted_labels = self.label_encoder.inverse_transform(y_pred)
        predicted_state_parameters = [complex(item[0],item[1]) for item in predictions[1]]
        return predicted_labels, predicted_state_parameters
