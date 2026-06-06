"""
This file contains the abstract base class for distillation models.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseRetrievalModel(ABC):
    @abstractmethod
    def __init__(self, **params):
        """
        Initialize the distillation model.

        Args:
            params (dict): Parameters for model configuration.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the distillation model.

        Args:
            data_folder (str): Path to the folder containing training data.
            save_folder (str): Path to save model parameters, weights and more.
            tb_folder (str): Path to save TensorBoard logs.
            params (dict): Dictionary containing training parameters such as:
                - lr (float): Learning rate for the optimizer.
                - batch_size (int): Batch size for training.
                - epochs (int): Number of training epochs.
                - percentage (float): Percentage of data to use in each epoch.
                - val_split (float): Validation split ratio.
        """
        pass

    @abstractmethod
    def load_weights(self, weights_path: str):
        """
        Load pre-trained weights into the model.

        Args:
            weights_path (str): Path to the file containing model weights.
        """
        pass

    @abstractmethod
    def predict(self, sentences: list, batch_size: int, verbose: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences (list): List of input sentences.
            batch_size (int): Batch size for processing.
            verbose (bool, optional): Whether to print progress. Defaults to False.

        Returns:
            np.ndarray: Array of embeddings for the input sentences.
        """
        pass