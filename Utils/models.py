from typing import List
import torch
import numpy as np
from matplotlib.pyplot import savefig

import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import torch.distributions as tdist
from torch.utils.data import DataLoader
from Utils.dataloader import DataLoaderGenerator
from Utils.lossfunctions import ELBOLoss, MSELoss
from Utils.processing import same_padding, force_padding
import os


# class BaseModel(nn.Module):
#     def __init__(self, config):
#         super(BaseModel, self).__init__()
#         self.config = config

# class VAEmodel(BaseModel):
class VAE(nn.Module):
    def __init__(self, input_dims:int, latent_dims:int = 6, n_channels:int = 1, n_kernels:int = 512, optimizer:optim.Optimizer = None, criterion:nn.Module = None, device:torch.device = None, normalized:bool = True):
        """
        Constructor for the VAE model.
        Args:
            input_dims (int): Number of input dimensions.
            latent_dims (int): Number of latent dimensions.
                Default is 6.
            n_channels (int): Number of channels in the input data.
                Default is 1.
            n_kernels (int): Number of kernels in the model. This scales as the encoder and decoder progress.
                Default is 512.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
                Default is Adam with learning rate 4e-4 and betas (0.9, 0.95).
            criterion (nn.Module): Loss function for the model.
                Default is ELBOLoss.
            device (torch.device): Device to be used for training.
                Default is 'cuda' if available, else 'cpu'.
            normalized (bool): Whether the input data is normalized
                Default is True.
        """
        super(VAE, self).__init__()
        if input_dims is None:
            raise ValueError("Input dimensions must be specified")
        if latent_dims is None:
            raise ValueError("Latent dimensions must be specified")
        
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.normalized = normalized
        self.build_model(n_channels=n_channels, n_kernels=n_kernels)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=4e-4, betas=(0.9, 0.95))

        if criterion is None:
            self.criterion = ELBOLoss()
        else:
            self.criterion = criterion

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_model(self, n_kernels:int = 512, n_channels:int = 1):
        """
        Function to build the VAE model. This function builds the encoder, decoder, and latent space. 
        The encoder and decoder are convolutional neural networks. The latent space is a multivariate normal distribution. 
        Args:
            n_kernels (int): Number of kernels in the model. This scales as the encoder and decoder progress.
                Default is 512.
            n_channels (int): Number of channels in the input data.
                Default is 1.
        """
        diff = (self.input_dims - self.latent_dims * 4)//4
        if diff < 0:
            raise ValueError("The latent space is too large for the input size.")
        if diff % 2 != 0:   # Maybe remove this
            diff += 1
        
        if self.normalized:
            # Encoder Structure:
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, n_kernels // 16, kernel_size=3, stride=2, # Shape (1, 100)
                        padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),   # Removed same_padding(self.input_dims, 3, 2)
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 16),
                nn.Conv1d(n_kernels // 16, n_kernels // 8, kernel_size=3, stride=2, # Shape (32, 80)
                        padding=force_padding(self.input_dims - diff, self.input_dims - diff*2,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 8),
                nn.Conv1d(n_kernels // 8, n_kernels // 4, kernel_size=3, stride=2, # Shape (64, 60)
                        padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 4),
                nn.Conv1d(n_kernels // 4, n_kernels, kernel_size=4, stride=2,   # Shape (128, 40)
                        padding=force_padding(self.input_dims - diff*3, self.latent_dims * 4, kernel_size=4, stride=2)),
                nn.LeakyReLU(), # Shape (512, 24)
                nn.Flatten(),    # Flatten the output to a 1D tensor
                nn.Linear(n_kernels * self.latent_dims * 4, self.latent_dims * 4),
                nn.LeakyReLU()
            )

            # Decoder Structure:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dims, self.latent_dims * 4),
                nn.LeakyReLU(),
                nn.Linear(self.latent_dims * 4, n_kernels * self.latent_dims * 4),
                nn.LeakyReLU(),
                nn.Unflatten(1, (n_kernels, self.latent_dims * 4)), # Shape (512, 24)
                nn.ConvTranspose1d(n_kernels, n_kernels // 4, kernel_size=4, stride=2,  # Shape (128, 40)
                                padding=force_padding(self.input_dims - diff*3, self.latent_dims * 4,kernel_size=4,stride=2)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 4),
                nn.ConvTranspose1d(n_kernels // 4, n_kernels // 8, kernel_size=3, stride=2, output_padding=1, # Shape (32, 60)  # output_padding=1 to fix the output size
                                padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 8),
                nn.ConvTranspose1d(n_kernels // 8, n_kernels // 16, kernel_size=3, stride=2, output_padding=1,   # Shape (8, 80)
                                padding=force_padding(self.input_dims - diff*1, self.input_dims - diff*2,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.BatchNorm1d(n_kernels // 16),
                nn.ConvTranspose1d(n_kernels // 16, n_channels, kernel_size=3, stride=2, output_padding=1,   # Shape (1, 100)
                                padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),
                # nn.Sigmoid()
                # Shape (1, 100)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, n_kernels // 16, kernel_size=3, stride=2, # Shape (1, 100)
                        padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),   # Removed same_padding(self.input_dims, 3, 2)
                nn.LeakyReLU(),
                nn.Conv1d(n_kernels // 16, n_kernels // 8, kernel_size=3, stride=2, # Shape (32, 80)
                        padding=force_padding(self.input_dims - diff, self.input_dims - diff*2,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.Conv1d(n_kernels // 8, n_kernels // 4, kernel_size=3, stride=2, # Shape (64, 60)
                        padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.Conv1d(n_kernels // 4, n_kernels, kernel_size=4, stride=2,   # Shape (128, 40)
                        padding=force_padding(self.input_dims - diff*3, self.latent_dims * 4, kernel_size=4, stride=2)),
                nn.LeakyReLU(), # Shape (512, 24)
                nn.Flatten(),    # Flatten the output to a 1D tensor
                nn.Linear(n_kernels * self.latent_dims * 4, self.latent_dims * 4),
                nn.LeakyReLU()
            )

            # Decoder Structure:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dims, self.latent_dims * 4),
                nn.LeakyReLU(),
                nn.Linear(self.latent_dims * 4, n_kernels * self.latent_dims * 4),
                nn.LeakyReLU(),
                nn.Unflatten(1, (n_kernels, self.latent_dims * 4)), # Shape (512, 24)
                nn.ConvTranspose1d(n_kernels, n_kernels // 4, kernel_size=4, stride=2,  # Shape (128, 40)
                                padding=force_padding(self.input_dims - diff*3, self.latent_dims * 4,kernel_size=4,stride=2)),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(n_kernels // 4, n_kernels // 8, kernel_size=3, stride=2, output_padding=1, # Shape (32, 60)  # output_padding=1 to fix the output size
                                padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(n_kernels // 8, n_kernels // 16, kernel_size=3, stride=2, output_padding=1,   # Shape (8, 80)
                                padding=force_padding(self.input_dims - diff*1, self.input_dims - diff*2,kernel_size=3,stride=2)),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(n_kernels // 16, n_channels, kernel_size=3, stride=2, output_padding=1,   # Shape (1, 100)
                                padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),
                # nn.Sigmoid()
                # Shape (1, 100)
            )

        # Latent Space:
        # Could change the shape. Maybe add another linear layer before the mean and std_dev layers.
        self.code_mean = nn.Linear(self.latent_dims * 4, self.latent_dims)
        self.code_std_dev = nn.Linear(self.latent_dims * 4, self.latent_dims)
        #self.code_std_dev.bias.data += sigma2_offset


    def forward(self, x):
        encoded_signal = self.encoder(x)    # Encode the input signal into the latent space
        encoded_signal = encoded_signal.view(encoded_signal.size(0), -1)

        code_mean = self.code_mean(encoded_signal)  # Get the mean of the latent space
        code_std_dev = self.code_std_dev(encoded_signal)    # Get the standard deviation of the latent space
        code_std_dev = torch.relu(code_std_dev) + 1e-2      # Bias the std
        mvn = tdist.MultivariateNormal(code_mean, torch.diag_embed(code_std_dev)) # Create a multivariate normal distribution
        code_sample = mvn.sample()  # Sample from the distribution
        #decoded = self.decoder(code_sample.unsqueeze(-1).unsqueeze(-1)) # Decode the sample
        decoded = self.decoder(code_sample) # Decode the sample
        return code_mean, code_std_dev, code_sample, decoded

    def train_model(self, train_loader:DataLoader, n_epochs:int=1, optimizer:optim.Optimizer=None, criterion:nn.Module=None, device:torch.device=None, verbose:bool=True):
        """
        Function to train the VAE model.
        Args:
            train_loader (DataLoader): DataLoader object for the training data.
            n_epochs (int): Number of epochs to train the model.
                Default is 1.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
                Default is None.
            criterion (nn.Module): Loss function for the model.
                Default is None.
            device (torch.device): Device to be used for training.
                Default is None.
            verbose (bool): Whether to print the loss at each epoch.
                Default is True.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        if criterion is not None:
            self.criterion = criterion
        if device is not None:
            self.device = device
        
        self.to(self.device)
        self.train()    # Could move this if recording validation loss
        train_loss = 0
        for epoch in range(n_epochs):
            for data, _ in train_loader:
                #batch.to(self.device)   # Move the batch to the device
                data = data.to(self.device)
                output = self(data)    # Get the output from the model
                loss = self.criterion(output, data, "train")   # Calculate the loss
                self.optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Computer the gradients
                self.optimizer.step()        # Update the weights
            self.criterion.loss_tracker_epoch_update()
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}")

    def encode_data(self, dataloader:DataLoader, device:torch.device=None):
        """
        Function to encode data using the VAE model. 
        Takes a DataLoader object and returns the encoded data in the latent space.
        Args:
            dataloader (DataLoader): DataLoader object for the data.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        if device is not None:
            self.device = device
        self.to(self.device)
        self.eval()
        embeddings = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self(data)
                embeddings.append(output[2].cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def save_model(self, path:str):
        """
        Function to save the model. Creates folders if they do not exist.
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), path)

    def load_model(self, path:str):
        self.load_state_dict(torch.load(path))

    def plot_loss(self, path:str=None):
        """
        Function to plot the loss of the model. If path is specified, the plot is saved to the path.
        """
        plt.figure()
        self.criterion.loss_tracker.plot_losses()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(path)
        else:
            plt.show()





class LSTM(nn.Module):
    def __init__(self, latent_dims = 6, n_neurons: int = 64, optimizer:optim.Optimizer = None, criterion:nn.Module = None, device:torch.device = None):
        """
        Constructor for the LSTM model.
        Args:
            latent_dims (int): Number of dimensions in the latent space.
                Default is 6.
            n_neurons (int): Number of neurons in the LSTM layers.
                Default is 64.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
                Default is None.
            criterion (nn.Module): Loss function for the model.
                Default is None.
            device (torch.device): Device to be used for training.
                Default is 'cuda' if available, else 'cpu'.
        """
        super(LSTM, self).__init__()
        self.latent_dims = latent_dims

        self.build_model(code_size=latent_dims, n_neurons=n_neurons)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

        if criterion is None:
            self.criterion = MSELoss()
        else:
            self.criterion = criterion

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_model(self, n_neurons: int = 64, code_size: int = 6):
        """
        Function to build the LSTM model.
        Args:
            n_neurons (int): Number of neurons in the LSTM layers.
                Default is 64.
            code_size (int): Number of dimensions in the latent space.
                Default is 6.
        """
        self.models = nn.ModuleList([
            nn.LSTM(code_size, n_neurons, batch_first=True),
            nn.LSTM(n_neurons, n_neurons, batch_first=True),
            nn.LSTM(n_neurons, code_size, batch_first=True),
            nn.Linear(code_size, code_size) # Add a linear layer to remove affect of previous layers activation function
        ])

        self.lstm_hidden = []
        for model in self.models:
            if isinstance(model, nn.LSTM):
                self.lstm_hidden.append(model.hidden_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        if len(x.shape) > 2:    # If the input is a batch
            h = [torch.zeros(1, x.size(0), hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
            c = [torch.zeros(1, x.size(0), hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
        else:   # If the input is a single sample
            h = [torch.zeros(1, hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
            c = [torch.zeros(1, hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]

        h = [torch.nn.init.xavier_normal_(h_) for h_ in h]
        c = [torch.nn.init.xavier_normal_(c_) for c_ in c]

        for model, h_, c_ in zip(self.models, h, c):
            if isinstance(model, nn.LSTM):
                x, (h_,c_) = model(x,(h_,c_))    # Get the output and hidden states
            else:
                x = model(x)
        return x
    
    def train_model(self, train_loader:DataLoader, n_epochs:int=1, optimizer:optim.Optimizer=None, criterion:nn.Module=None, device:torch.device=None, verbose:bool=True):
        """
        Function to train the LSTM model.
        Args:
            train_loader (DataLoader): DataLoader object for the training data.
            n_epochs (int): Number of epochs to train the model.
                Default is 1.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
                Default is None.
            criterion (nn.Module): Loss function for the model.
                Default is None.
            device (torch.device): Device to be used for training.
                Default is None.
            verbose (bool): Whether to print the loss at each epoch.
                Default is True.
        """
        if optimizer is not None:
            self.optimizer = optimizer
        if criterion is not None:
            self.criterion = criterion
        if device is not None:
            self.device = device
        
        self.to(self.device)
        self.train()    # Could move this if recording validation loss
        train_loss = 0
        for epoch in range(n_epochs):
            for data, _ in train_loader:
                data = data.to(self.device)   # Move the batch to the device
                output = self(data)    # Get the output from the model
                loss = self.criterion(output, data, 'train')   # Calculate the loss
                self.optimizer.zero_grad()   # Zero the gradients
                loss.backward()         # Computer the gradients
                self.optimizer.step()        # Update the weights
            self.criterion.loss_tracker_epoch_update()
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}")

    def save_model(self, path:str):
        """
        Function to save the model. Creates folders if they do not exist.
        """
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), path)

    def load_model(self, path:str):
        self.load_state_dict(torch.load(path))

    def plot_loss(self, path:str=None):
        """
        Function to plot the loss of the model. If path is specified, the plot is saved to the path.
        """
        plt.figure()
        self.criterion.loss_tracker.plot_losses()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            savefig(path)
        else:
            plt.show()

class VAE_LSTM(nn.Module):
    def __init__(self, vae:VAE=None, lstm:LSTM=None, input_dims:int = None, latent_dims:int = None, n_channels:int = 1, n_kernels_vae:int = 512, n_neurons_lstm:int = 64, optimizer_vae:optim.Optimizer = None, optimizer_lstm:optim.Optimizer = None, criterion_vae:nn.Module = None, criterion_lstm:nn.Module = None, device:torch.device = None, normalized:bool = True):
        
        super(VAE_LSTM, self).__init__()
        if vae is not None:
            self.vae = vae
        else:
            self.vae = VAE(input_dims, latent_dims, n_channels, n_kernels_vae, optimizer_vae, criterion_vae, device, normalized)
        self.input_dims = self.vae.latent_dims
        self.latent_dims = self.vae.latent_dims

        if lstm is not None:
            self.lstm = lstm
        else:
            self.lstm = LSTM(self.latent_dims, n_neurons_lstm, optimizer_lstm, criterion_lstm, device)
        
    def train_model(self, train_loader:DataLoader, n_epochs:int=1, optimizer:optim.Optimizer=None, criterion:nn.Module=None, device:torch.device=None, verbose:bool=True):
        self.vae.train_model(train_loader, n_epochs, optimizer, criterion, device, verbose)
        
        encoded_training_data = self.vae.encode_data(train_loader)
        lstm_dataloader = DataLoaderGenerator(encoded_training_data, batch_size=train_loader.batch_size)
        train_loader_lstm = lstm_dataloader.generate()

        self.lstm.train_model(train_loader_lstm, n_epochs, optimizer, criterion, device, verbose)

    def save_model(self, path:str):
        self.vae.save_model(path + "_vae")
        self.lstm.save_model(path + "_lstm")

    def load_model(self, path:str):
        self.vae.load_model(path + "_vae")
        self.lstm.load_model(path + "_lstm")
    
    def load_vae_model(self, path:str):
        self.vae.load_model(path)

    def load_lstm_model(self, path:str):
        self.lstm.load_model(path)

    def plot_loss(self, path:str=None):
        if path is None:
            self.vae.plot_loss()
            self.lstm.plot_loss()
        else:
            self.vae.plot_loss(path + "_vae")
            self.lstm.plot_loss(path + "_lstm")

    def evaluate(self, dataloader:DataLoader, threshold, labels=None, device:torch.device=None):
        """
        Method to evaluate the model on a dataset.
        """
        # Need to add metrics such as Precision, Recall, and F1 score
        # Method for if labels given or not.
        # Find best threshold for anomaly detection
        # Return windows with anomalies. Maybe add in way to unroll windows.

        if device is not None:
            self.device = device

        self.to(self.device)
        self.eval()
        #embeddings = []
        with torch.no_grad():
            for input, _ in dataloader:
                input = input.to(self.device)
                vae_output = self.vae(input)
                lstm_output = self.lstm(vae_output[2])
                output = self.vae.decoder(lstm_output)
                

