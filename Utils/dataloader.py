from copy import deepcopy
from typing import Dict, Optional, Tuple
from pandas import DataFrame
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class DataLoaderGenerator:
    """
    Class to generate DataLoader objects for training and validation data.
    """
    def __init__(
            self, 
            data=None, 
            batch_size:int=32,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = torch.device('cpu')
            ):
        """
        Args:
            data: Data to be used for training and validation.
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
            self.data_loader_params = {
            'shuffle': True,    # The dataloader will shuffle its outputs at each epoch
            'num_workers': 0,   # The number of workers that the dataloader will use to generate the batches
            'drop_last': True,  # Drop the last batch if it is smaller than the batch size
            }
        else:
            self.data_loader_params = deepcopy(data_loader_params)
            

    def generate(
            self, 
            data=None, 
            batch_size:int=-1,
            data_loader_params:Optional[Dict] = None,
            device: torch.device = None
            ) -> Tuple[DataLoader, DataLoader]:
        """
        Generate DataLoader objects for training and validation data.
        Args:
            data: Data to be used for training and validation.
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

        # # Split data into training and validation data (ONLY RANDOM SPLIT IS IMPLEMENTED)
        # if isinstance(self.data, DataFrame):
        #     train_data = self.data.sample(frac=0.9, random_state=0)
        #     val_data = self.data.drop(train_data.index)
        # else:
        #     train_indx = np.random.choice(range(len(self.data)), int(0.9 * len(self.data)), replace=False)
        #     val_indx = np.setdiff1d(range(len(self.data)), train_indx)
        #     train_data = self.data[train_indx]
        #     val_data = self.data[val_indx]

        # Convert data to numpy arrays
        if isinstance(self.data, DataFrame):
            train_data = self.data.to_numpy()
        elif isinstance(self.data, np.ndarray):
            train_data = self.data
        else:
            train_data = np.array(self.data)

        train_data = train_data.astype(np.float32)
        #val_data = val_data.astype(np.float32)

        if len(train_data.shape) == 2:
            train_data = train_data[:,np.newaxis,:] # Add channel dimension
            # val_data = val_data[:,np.newaxis,:] # Add channel dimension

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(train_data).to(self.device),
            torch.tensor(train_data).to(self.device)
        )
        # val_dataset = TensorDataset(
        #     torch.tensor(val_data).to(self.device),
        #     torch.tensor(val_data).to(self.device)
        # )

        # Generate DataLoader objects for training and validation data
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, **self.data_loader_params)
        #val_loader = DataLoader(val_dataset, batch_size=self.batch_size, **self.data_loader_params)

        return train_loader#, val_loader


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    #data = pd.DataFrame(np.array([range(100)]).T)
    # data = np.array(range(100))
    data = np.random.rand(100, 2)
    data_loader_generator = DataLoaderGenerator(data, batch_size=32)
    train_loader, val_loader = data_loader_generator.generate()
    print(train_loader)
    print(val_loader)