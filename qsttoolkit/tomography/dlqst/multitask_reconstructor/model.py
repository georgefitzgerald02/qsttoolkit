import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from qsttoolkit.tomography.dlqst.multitask_reconstructor.architecture import build_multitask_reconstructor


class MultitaskQuantumStateTomography:
    """
    A class for training and evaluating a multitask regression/classification model for quantum state tomography.

    Attributes
    ----------
    dim : int
        The Hilbert space dimensionality.
    latent_dim : int
        The phase space dimensionality.
    X_train : np.ndarray
        The training data.
    X_test : np.ndarray
        The test data.
    y_train : dict
        A dictionary containing both the training classification and regression labels.
    y_test : dict
        A dictionary containing both the test classification and regression labels.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder for the classification labels.
    model : tf.keras.Model
        The multitask model.
    early_stopping : tf.keras.callbacks.EarlyStopping
        The early stopping callback.
    lr_scheduler : tf.keras.callbacks.ReduceLROnPlateau
        The learning rate scheduler callback.
    callbacks : list
        The list of callbacks. 
    history : tf.keras.callbacks.History
        The training history.
    """
    def __init__(self, dim: int, X_train, X_test, y_train, y_test, label_encoder, early_stopping_patience=3, lr_scheduler_factor=0.5, lr_scheduler_patience=2):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.dim = dim
        self.latent_dim = X_train[0].shape[0]

        self.model = build_multitask_reconstructor(input_shape=(self.latent_dim, self.latent_dim, 1), num_classes=len(self.label_encoder.classes_), num_regression_outputs=2)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_scheduler_factor, patience=lr_scheduler_patience)
        self.callbacks = [self.early_stopping, self.lr_scheduler]

    def train(self, optimizer='adam', classification_loss='sparse_categorical_crossentropy', regression_loss='mse', classification_loss_weight=1.0, regression_loss_weight=0.5, classification_metric='accuracy', regression_metric='mae', epochs=30, batch_size=32, validation_split=0.2):
        """
        Compiles and trains the model.

        Parameters
        ----------
        optimizer : str
            The optimizer to use in the training. Defaults to 'adam'.
        classification_loss : str
            The classification loss function to use in the training. Defaults to 'sparse_categorical_crossentropy'.
        regression_loss : str
            The regression loss function to use in the training. Defaults to 'mse'.
        classification_loss_weight : float
            The weight of the classification loss in the total loss. Defaults to 1.0.
        regression_loss_weight : float
            The weight of the regression loss in the total loss. Defaults to 0.5.
        classification_metric : str
            The metric to measure classification performance during training. Defaults to 'accuracy'.
        regression_metric : str
            The metric to measure regression performance during training. Defaults to 'mae'.
        epochs : int
            The number of epochs to train the model. Defaults to 30.
        batch_size : int
            The training batch size. Defaults to 32.
        validation_split : float
            The fraction of the training data to use as validation data. Defaults to 0.2.
        """
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
                                      callbacks=self.callbacks)

    def plot_training(self):
        """
        Plots the training history over epochs.
        """
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        # Plot training & validation accuracy
        axs[0].plot(self.history.history['classification_output_accuracy'], label='train accuracy')
        axs[0].plot(self.history.history['val_classification_output_accuracy'], label='val accuracy')
        axs[0].set_ylim(0,1)
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend()

        # Plot training & validation mae
        axs[1].plot(self.history.history['regression_output_mae'], label='train mae')
        axs[1].plot(self.history.history['val_regression_output_mae'], label='val mae')
        axs[1].set_title('Model MAE')
        axs[1].set_ylabel('MAE')
        axs[1].set_xlabel('Epoch')
        axs[1].legend()

        # Plot training & validation loss
        axs[2].plot(self.history.history['loss'], label='train loss')
        axs[2].plot(self.history.history['val_loss'], label='val loss')
        axs[2].set_title('Model Loss')
        axs[2].set_ylabel('Loss')
        axs[2].set_xlabel('Epoch')
        axs[2].legend()

        plt.show()

    def evaluate_classification(self, include_confusion_matrix=True, include_classification_report=True):
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
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

        if include_classification_report:
            # Classification report
            class_report = classification_report(self.y_test['classification_output'], y_pred)
            print("Classification Report:")
            print(class_report)

    def evaluate_regression(self):
        """
        Evaluates the regression performance of the model.
        """
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

    def infer(self, data):
        """
        Infers the quantum state label and key parameter from input data.

        Parameters
        ----------
        data : np.ndarray
            The input data.
        """
        predictions = self.model.predict(data)
        y_pred = np.argmax(predictions[0], axis=1)
        predicted_labels = self.label_encoder.inverse_transform(y_pred)
        predicted_state_parameters = [complex(item[0],item[1]) for item in predictions[1]]
        return predicted_labels, predicted_state_parameters
