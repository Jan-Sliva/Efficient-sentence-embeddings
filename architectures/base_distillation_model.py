"""
This file contains the abstract base class for distillation models.
"""

from abc import ABC, abstractmethod
import torch

class BaseDistillationModel(ABC):
    @abstractmethod
    def __init__(self, emb_path: str, **kwargs):
        """
        Initialize the distillation model.

        Args:
            emb_path (str): Path to the embeddings file.
            **kwargs: Additional keyword arguments for model configuration.
        """
        pass

    @abstractmethod
    def train(self, data_folder: str, save_folder: str, tb_folder: str, 
              lr: float, batch_size: int, epochs: int = 1, percentage: float = 1, val_split: float = 0.1):
        """
        Train the distillation model.

        Args:
            data_folder (str): Path to the folder containing training data.
            save_folder (str): Path to save the trained model checkpoints.
            tb_folder (str): Path to save TensorBoard logs.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            percentage (float, optional): Percentage of data to use in each epoch. Defaults to 1.
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
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform inference using the trained model.

        Args:
            input_tensor (torch.Tensor): Input tensor for inference.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        pass