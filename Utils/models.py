from typing import List
import torch
import numpy as np
from matplotlib.pyplot import savefig

import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import torch.distributions as tdist
from torch.utils.data import DataLoader

from Utils.lossfunctions import ELBOLoss, MSELoss
from Utils.processing import same_padding, force_padding
import os


# class BaseModel(nn.Module):
#     def __init__(self, config):
#         super(BaseModel, self).__init__()
#         self.config = config

# class VAEmodel(BaseModel):
class VAE(nn.Module):
    def __init__(self, input_dims, latent_dims = 6, optimizer:optim.Optimizer = None, criterion:nn.Module = None, device:torch.device = None):
    # def __init__(self, config):
        # super(VAEmodel, self).__init__(config)
        super(VAE, self).__init__()
        #self.input_dims = self.config['l_win'] * self.config['n_channel']
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.build_model()

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

    def build_model(self, n_kernels:int = 512, n_channels:int = 1, sigma2_offset:float = 1e-2):
        # init = nn.init.xavier_uniform_

        diff = (self.input_dims - self.latent_dims * 4)//4
        if diff < 0:
            raise ValueError("The latent space is too large for the input size.")
        if diff % 2 != 0:   # Maybe remove this
            diff += 1
        
        # Encoder Structure:
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_kernels // 16, kernel_size=3, stride=2, # Shape (1, 100)
                      padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),   # Removed same_padding(self.input_dims, 3, 2)
            nn.LeakyReLU(),
            #nn.BatchNorm1d(n_kernels // 16),
            nn.Conv1d(n_kernels // 16, n_kernels // 8, kernel_size=3, stride=2, # Shape (32, 80)
                      padding=force_padding(self.input_dims - diff, self.input_dims - diff*2,kernel_size=3,stride=2)),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(n_kernels // 8),
            nn.Conv1d(n_kernels // 8, n_kernels // 4, kernel_size=3, stride=2, # Shape (64, 60)
                      padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(n_kernels // 4),
            nn.Conv1d(n_kernels // 4, n_kernels, kernel_size=4, stride=2,   # Shape (128, 40)
                      padding=force_padding(self.input_dims - diff*3, self.latent_dims * 4, kernel_size=4, stride=2)),
            nn.LeakyReLU(), # Shape (512, 24)
            nn.Flatten(),    # Flatten the output to a 1D tensor
            nn.Linear(n_kernels * self.latent_dims * 4, self.latent_dims * 4),
            nn.LeakyReLU()
        )

        # Latent Space:
        # Could change the shape. Maybe add another linear layer before the mean and std_dev layers.
        self.code_mean = nn.Linear(self.latent_dims * 4, self.latent_dims)
        self.code_std_dev = nn.Linear(self.latent_dims * 4, self.latent_dims)
        #self.code_std_dev.bias.data += sigma2_offset

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
            #nn.BatchNorm1d(n_kernels // 4),
            nn.ConvTranspose1d(n_kernels // 4, n_kernels // 8, kernel_size=3, stride=2, output_padding=1, # Shape (32, 60)  # output_padding=1 to fix the output size
                               padding=force_padding(self.input_dims - diff*2, self.input_dims - diff*3,kernel_size=3,stride=2)),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(n_kernels // 8),
            nn.ConvTranspose1d(n_kernels // 8, n_kernels // 16, kernel_size=3, stride=2, output_padding=1,   # Shape (8, 80)
                               padding=force_padding(self.input_dims - diff*1, self.input_dims - diff*2,kernel_size=3,stride=2)),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(n_kernels // 16),
            nn.ConvTranspose1d(n_kernels // 16, n_channels, kernel_size=3, stride=2, output_padding=1,   # Shape (1, 100)
                               padding=force_padding(self.input_dims, self.input_dims - diff,kernel_size=3,stride=2)),
            # nn.Sigmoid()
            # Shape (1, 100)
        )

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
    
    # def train_model(self, data, config, optimizer, criterion, cp_callback):
    #     for epoch in range(config['num_epochs_vae']):
    #         for x in data.train_loader:
    #             optimizer.zero_grad()
    #             output = self(x)
    #             loss = criterion(output[3], x)
    #             loss.backward()
    #             optimizer.step()
    #         print(f"Epoch {epoch+1}/{config['num_epochs_vae']}, Loss: {loss.item()}")
    #     torch.save(self.state_dict(), cp_callback)

    def train_model(self, train_loader:DataLoader, n_epochs:int=1, optimizer:optim.Optimizer=None, criterion:nn.Module=None, device:torch.device=None, verbose:bool=True):
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
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}")

    def encode_data(self, dataloader:DataLoader, device:torch.device=None):
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
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), path)

    def load_model(self, path:str):
        self.load_state_dict(torch.load(path))





class LSTM(nn.Module):
    def __init__(self, latent_dims = 6, optimizer:optim.Optimizer = None, criterion:nn.Module = None, device:torch.device = None):
        super(LSTM, self).__init__()
        self.latent_dims = latent_dims

        self.build_model(code_size=latent_dims)

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
        # self.model = nn.Sequential(
        #     nn.LSTM(code_size, n_neurons, batch_first=True),
        #     nn.LSTM(n_neurons, n_neurons, batch_first=True),
        #     nn.LSTM(n_neurons, code_size, batch_first=True),
        #     nn.Linear(code_size, code_size) # Add a linear layer to remove affect of previous layers activation function
        # )
        self.models = nn.ModuleList([
            nn.LSTM(code_size, n_neurons, batch_first=True),
            nn.LSTM(n_neurons, n_neurons, batch_first=True),
            nn.LSTM(n_neurons, code_size, batch_first=True),
            nn.Linear(code_size, code_size) # Add a linear layer to remove affect of previous layers activation function
        ])

        # for model in self.models:
        #     for name, param in model.named_parameters():
        #         if 'weight' in name:
        #             nn.init.xavier_normal_(param)
        #         else:
        #             nn.init.zeros_(param)
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
            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}")

    def save_model(self, path:str):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), path)

    def load_model(self, path:str):
        self.load_state_dict(torch.load(path))
