from copy import deepcopy
from typing import Dict, Optional, Tuple
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader, TensorDataset

class DataLoaderGenerator:
    """
    Class to generate DataLoader objects for training and validation data.
    """
    def __init__(
            self, 
            data:DataFrame=None, 
            batch_size:int=32,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = torch.device('cpu')
            ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            data (DataFrame): Data to be used for training and validation.
            batch_size (int): Batch size for DataLoader objects.
            data_loader_params (Dict): Additional parameters for DataLoader objects.
            device (torch.device): Device to be used for training.
        Returns:
            Tuple[DataLoader, DataLoader]: DataLoader objects for training and validation data.
        """
        self.data = data
        self.batch_size = batch_size
        self.data_loader_params = data_loader_params
        self.device = device

        if self.data_loader_params is None:
            self.data_loader_params = {}
        else:
            self.data_loader_params = deepcopy(data_loader_params)

        if data is not None:
            return self.generate(data, batch_size, data_loader_params, device)
        else:
            return None, None   # Return None if data is not provided

    def generate(
            self, 
            data:DataFrame=None, 
            batch_size:int=-1,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = None
            ) -> Tuple[DataLoader, DataLoader]:
        """
        Generate DataLoader objects for training and validation data.
        Args:
            data (DataFrame): Data to be used for training and validation.
            batch_size (int): Batch size for DataLoader objects.
            data_loader_params (Dict): Additional parameters for DataLoader objects.
            device (torch.device): Device to be used for training.
        Returns:
            Tuple[DataLoader, DataLoader]: DataLoader objects for training and validation data.
        """
        if data is not None:
            self.data = data
        if batch_size != -1:
            self.batch_size = batch_size
        if data_loader_params is not None:
            self.data_loader_params = deepcopy(data_loader_params)
        if device is not None:
            self.device = device

        # Split data into training and validation data (ONLY RANDOM SPLIT IS IMPLEMENTED)
        train_data = self.data.sample(frac=0.9, random_state=0)
        val_data = self.data.drop(train_data.index)

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(train_data).to(self.device),
            torch.tensor(train_data).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_data).to(self.device),
            torch.tensor(val_data).to(self.device)
        )

        # Generate DataLoader objects for training and validation data
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, **self.data_loader_params)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, **self.data_loader_params)

        return train_loader, val_loader

