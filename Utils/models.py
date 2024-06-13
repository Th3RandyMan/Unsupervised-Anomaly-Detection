from typing import List
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import matplotlib.lines as mlines
import matplotlib.pylab as plt
import torch.distributions as tdist
from torch.utils.data import DataLoader
from Utils.dataloader import DataLoaderGenerator
from Utils.lossfunctions import KLDLoss, SumLoss, MSELoss
from Utils.processing import same_padding, force_padding
import os


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
                Default is sum of KLD and BCE.
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
            #self.criterion = SumLoss([KLDLoss(), MSELoss()])   # Has not worked well...
            self.criterion = MSELoss()
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
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}", flush=True)

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

    def plot_loss(self, path:str=None, getfig:bool=False):
        """
        Function to plot the loss of the model. If path is specified, the plot is saved to the path.
        """
        fig = plt.figure()
        self.criterion.loss_tracker.plot_losses()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(path)
        elif getfig:
            return fig
        else:
            plt.show()

    # def get_reconstruction_error(self, test_data:np.ndarray, device:torch.device=None):
    #     """
    #     Method to evaluate the model on a dataset.
    #     Args:
    #         test_data (np.ndarray): Test data to be evaluated.
    #         device (torch.device): Device to be used for training.
    #             Default is None.
    #     """
    #     from Utils.processing import window_data

    #     if device is not None:
    #         self.device = device

    #     self.to(self.device)
    #     self.eval()

    #     test_windowed_data = window_data(test_data, window_size=self.input_dims, stride=1)
    #     test_windowed_data = torch.tensor(test_windowed_data).to(self.device)
    #     window_errors = torch.zeros(len(test_windowed_data))
    #     with torch.no_grad():
    #         for i, input in enumerate(test_windowed_data):  # Per Sample, not per batch
    #             output = self(input)
    #             window_errors[i] = torch.mean((input - output[-1])**2)
        
    #     reconstruction_error = torch.zeros(len(test_data))
    #     weights = torch.zeros(len(test_data))
    #     for i in range(len(test_data) - self.input_dims + 1):
    #         weights[i:i+self.input_dims] += 1
    #         reconstruction_error[i:i+self.input_dims] += window_errors[i]
    #     reconstruction_error /= weights # Average the reconstruction error over the windows
    #     return reconstruction_error

    def get_reconstruction_error(self, test_data:np.ndarray, device:torch.device=None):
        """
        Method to evaluate the model on a dataset.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        from Utils.processing import window_data
        BATCH_SIZE = 512

        if device is not None:
            self.device = device

        self.to(self.device)
        self.eval()

        test_windowed_data = window_data(test_data, window_size=self.input_dims, stride=1).astype(np.float32)
        if len(test_windowed_data.shape) == 2:
            test_windowed_data = test_windowed_data[:, np.newaxis, :]   # Add dim for channel
        shape = test_windowed_data.shape
        pad = BATCH_SIZE - shape[0] % BATCH_SIZE
        test_windowed_data = np.pad(test_windowed_data, ((0, pad), (0, 0), (0, 0)))
        test_windowed_data = test_windowed_data.reshape(-1, BATCH_SIZE, shape[1], shape[2])

        test_windowed_data = torch.tensor(test_windowed_data).to(self.device)
        window_errors = torch.zeros(shape[0]).to(self.device)

        with torch.no_grad():
            for i, input in enumerate(test_windowed_data):
                if i == len(test_windowed_data) - 1:    # If the last batch
                    input = input[:shape[0] % BATCH_SIZE]
                output = self(input)
                window_errors[i*BATCH_SIZE:i*BATCH_SIZE + len(input)] = torch.mean((input - output[-1])**2, dim=(1, 2))
        
        reconstruction_error = torch.zeros(len(test_data)).to(self.device)
        weights = torch.zeros(len(test_data)).to(self.device)
        for i in range(len(window_errors)):
            weights[i:i+self.input_dims] += 1
            reconstruction_error[i:i+self.input_dims] += window_errors[i]
        reconstruction_error /= weights # Average the reconstruction error over the windows
        return reconstruction_error
    
    def loc_anomalies(self, reconstruction_error:torch.Tensor, threshold:float=0.9, threshold_option:int=1):
        """
        Method to locate anomalies in the reconstruction error.
        Args:
            reconstruction_error (torch.Tensor): Reconstruction error from the model.
            threshold (float): Threshold for anomaly detection.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
        """
        #threshold = torch.tensor(threshold).to(reconstruction_error.device)
        if threshold_option == 1:
            # Option 1: Use the mean and standard deviation of the reconstruction error
            error_threshold = torch.mean(reconstruction_error) + threshold * torch.std(reconstruction_error)
            return torch.where(reconstruction_error > error_threshold)[0].to('cpu')
        elif threshold_option == 2:
            # Option 2: Use percentage between max and min
            error_threshold = threshold * (torch.max(reconstruction_error) - torch.min(reconstruction_error)) + torch.min(reconstruction_error)
            return torch.where(reconstruction_error > error_threshold)[0].to('cpu')
        else:
            raise ValueError("Threshold option must be 1 or 2.")

    def get_anomalies(self, test_data:np.ndarray, threshold:float=0.9, threshold_option:int=1, device:torch.device=None):
        """
        Method to get error from the model on a dataset.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
                If None, return the reconstruction error.
                If not None, return the indices of the anomalies.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        reconstruction_error = self.get_reconstruction_error(test_data, device)
        if threshold is None:
            return reconstruction_error
        else:
            return self.loc_anomalies(reconstruction_error, threshold, threshold_option)
        
    def get_anomaly_bool(self, test_data:np.ndarray, threshold:float=0.9, threshold_option:int=1, device:torch.device=None):
        """
        Method to get anomaly locations as a boolean array.
        Shares the same shape as the input data.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        anomaly_indices = self.get_anomalies(test_data, threshold, threshold_option, device)
        anomaly_bool = np.zeros(len(test_data), dtype=bool)
        anomaly_bool[anomaly_indices] = True
        return anomaly_bool
    
    def score_anomalies(self, detected_anomalies:torch.Tensor, true_anomalies:np.ndarray):
        """
        Get the precision, recall, and F1 score for the detected anomalies.
        Args:
            detected_anomalies (torch.Tensor): Anomalies detected by the model.
            true_anomalies (np.ndarray): True anomalies in the data.
        """
        true_positives = len(np.intersect1d(detected_anomalies, true_anomalies))
        false_positives = len(np.setdiff1d(detected_anomalies, true_anomalies))
        false_negatives = len(np.setdiff1d(true_anomalies, detected_anomalies))
        
        if true_positives + false_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        
        recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_score
    
    def augment_anomalies(self, detected_anomalies:torch.Tensor, true_anomalies:np.ndarray):
        """
        Method for reducing impact of consecutive windows with the same anomaly.
        Both detected anomalies and true anomalies are lists of indices.
        Args:
            detected_anomalies (torch.Tensor): Anomalies detected by the model.
            true_anomalies (np.ndarray): True anomalies in the data.
        Returns:
            torch.Tensor: Detected anomalies after discounting.
        """
        # extended_detected_anomalies = list(detected_anomalies)
        # for tru_anom in true_anomalies:
        #     for det_anom in detected_anomalies:
        #         if det_anom in tru_anom:
        #             original_det = set(extended_detected_anomalies)
        #             current_anom = set([tru_anom])
        #             extended_detected_anomalies += list(original_det - current_anom)

        extended_detected_anomalies = list(detected_anomalies)
        extended_detected_anomalies_set = set(extended_detected_anomalies)
        
        for true_anomaly in true_anomalies:
            if true_anomaly not in extended_detected_anomalies_set:
                extended_detected_anomalies.append(true_anomaly)
        
        extended_detected_anomalies.sort()
        
        return torch.tensor(extended_detected_anomalies).sort().values

        
    def evaluate(self, test_data:np.ndarray, anomalies:np.ndarray, threshold_option:int=1, device:torch.device=None, verbose:bool=True, plot:bool=False, path:str=None, getfig:bool=False):
        """
        Method to evaluate the model on a dataset. Requires the true anomalies.
        Attempts to find the best threshold for anomaly detection.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            anomalies (np.ndarray): True anomalies in the data.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
            verbose (bool): Whether to print the results.
                Default is True.
            plot (bool): Whether to plot the results.
                Default is False.
            path (str): Path to save the plot.
                Default is None.
            getfig (bool): Whether to return the figure.
        Returns:
            float: Best threshold for anomaly detection.
            float: Precision at the best threshold.
            float: Recall at the best threshold.
            float: F1 Score at the best threshold.
            float: Best threshold for augmented anomaly detection.
            float: Precision at the best threshold for augmented anomaly detection.
            float: Recall at the best threshold for augmented anomaly detection.
            float: F1 Score at the best threshold for augmented anomaly detection.
            fig: Figure object if getfig is True.
        """
        reconstruction_error = self.get_reconstruction_error(test_data, device)
        if anomalies is None:
            raise ValueError("Anomalies must be specified.")
        if isinstance(anomalies, list):
            anomalies = np.array(anomalies)
            
        if isinstance(anomalies, np.ndarray) and anomalies.dtype == bool and len(anomalies) == len(test_data):
            anom_loc = np.where(anomalies)[0]
        elif isinstance(anomalies, np.ndarray) and anomalies.dtype != np.int64 and len(anomalies) < len(test_data):
            anom_loc = anomalies
        else:
            raise ValueError("Anomalies must be a list or numpy array of indices or booleans.")
        anom_loc = torch.tensor(anom_loc)
        
        # threshold_list = np.linspace(0.1, 1, 10)
        # threshold_list = np.linspace(0.4, 1, 25)
        threshold_list = np.linspace(0.1, 1, 37)

        precisions = np.zeros(len(threshold_list))
        recalls = np.zeros(len(threshold_list))
        f1_scores = np.zeros(len(threshold_list))
        aug_precisions = np.zeros(len(threshold_list))
        aug_recalls = np.zeros(len(threshold_list))
        aug_f1_scores = np.zeros(len(threshold_list))
        for i, threshold in enumerate(threshold_list):
            detected_anomalies = self.loc_anomalies(reconstruction_error, threshold, threshold_option)
            precisions[i], recalls[i], f1_scores[i] = self.score_anomalies(detected_anomalies, anom_loc)
            aug_detected_anomalies = self.augment_anomalies(detected_anomalies, anom_loc)
            aug_precisions[i], aug_recalls[i], aug_f1_scores[i] = self.score_anomalies(aug_detected_anomalies, anom_loc)
            if verbose:
                print(f"Threshold: {threshold}, Precision: {precisions[i]}, Recall: {recalls[i]}, F1 Score: {f1_scores[i]}")
                print(f"\tAugmented Precision: {aug_precisions[i]}, Augmented Recall: {aug_recalls[i]}, Augmented F1 Score: {aug_f1_scores[i]}")

        best_index = np.argmax(f1_scores)
        best_index_aug = np.argmax(aug_f1_scores)
        if plot or path is not None or getfig:  
            fig = plt.figure(figsize=(12, 6))

            # Plotting the first subplot
            plt.subplot(1, 2, 1)
            plt.plot(threshold_list, precisions, label="Precision")
            plt.plot(threshold_list, recalls, label="Recall")
            plt.plot(threshold_list, f1_scores, label="F1 Score")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()

            # Plotting the second subplot
            plt.subplot(1, 2, 2)
            plt.plot(threshold_list, aug_precisions, label="Precision")
            plt.plot(threshold_list, aug_recalls, label="Recall")
            plt.plot(threshold_list, aug_f1_scores, label="F1 Score")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()

            if path is not None:
                folder = os.path.dirname(path)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                plt.savefig(path)
            elif getfig:
                return threshold_list[best_index], precisions[best_index], recalls[best_index], f1_scores[best_index], threshold_list[best_index_aug], aug_precisions[best_index_aug], aug_recalls[best_index_aug], aug_f1_scores[best_index_aug], fig
            else:
                plt.show()

        return threshold_list[best_index], precisions[best_index], recalls[best_index], f1_scores[best_index], threshold_list[best_index_aug], aug_precisions[best_index_aug], aug_recalls[best_index_aug], aug_f1_scores[best_index_aug]

    def plot_anomaly(self, test_data:np.ndarray, threshold, anomalies:np.ndarray=None, Fs=1, normalize:bool=True, threshold_option:int=1, device:torch.device=None, path:str=None, getfig:bool=False):
        """
        Method to plot the anomalies detected by the model.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
            anomalies (np.ndarray): True anomalies in the data. Boolean array.
            Fs (int): Sampling frequency of the data.
            normalize (bool): Whether to normalize the data before plotting.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
            path (str): Path to save the plot.
                Default is None.
            getfig (bool): Whether to return the figure.
        """
        if device is not None:
            self.device = device
        if normalize:
            show_data = test_data.copy()
            test_data = (test_data - np.mean(test_data)) / np.std(test_data)

        if anomalies is not None:
            if isinstance(anomalies, np.ndarray) and anomalies.dtype == bool and len(anomalies) == len(test_data):
                anom_loc = np.where(anomalies)[0]
            elif isinstance(anomalies, np.ndarray) and anomalies.dtype != np.int64 and len(anomalies) < len(test_data):
                anom_loc = anomalies
            else:
                raise ValueError("Anomalies must be a list or numpy array of indices or booleans.")
            anom_loc = torch.tensor(anom_loc)

        #detected_anomalies = self.get_anomaly_bool(test_data, threshold=threshold, threshold_option=threshold_option, device=self.device)
        detected_anomalies = self.get_anomalies(test_data, threshold=threshold, threshold_option=threshold_option, device=self.device)

        t = np.arange(len(test_data)) / Fs
        fig = plt.figure()
        # plt.plot(t, test_data)

        #plt.vlines(t[detected_anomalies], ymin=np.min(test_data), ymax=np.max(test_data), colors='g', linestyles='solid', label="Detected Anomalies", alpha=0.5, linewidth=1)

        if anomalies is not None:
           plt.vlines(t[anom_loc], ymin=np.min(test_data), ymax=np.max(test_data), colors='r', linestyles='solid', label="True Anomalies", alpha=0.5, linewidth=0.05)
        plt.vlines(t[detected_anomalies], ymin=np.min(test_data), ymax=np.max(test_data), colors='g', linestyles='solid', label="Detected Anomalies", alpha=0.5, linewidth=0.1)

        if normalize:
            test_data = show_data

        plt.plot(t, test_data, linewidth=2.5, color='black')
        plt.plot(t, test_data, label="Data", color='b')

        
        # plt.axvline(x=t[detected_anomalies[0]], color='g', linestyle='solid', label="Detected Anomalies")
        # for detected_anom in detected_anomalies[1:]:
        #     plt.axvline(x=t[detected_anom], color='g', linestyle='solid')

        # if anomalies is not None:
        #     #plt.scatter(t[anomalies], test_data[anomalies], color='r', label="True Anomalies")
        #     plt.axvline(x=t[anom_loc[0]], color='r', linestyle='--', label="True Anomalies") 
        #     for anom in anom_loc[1:]:
        #         plt.axvline(x=t[anom], color='r', linestyle='--')        #plt.plot(t, detected_anomalies, label="Detected Anomalies", color='g--')
        
        # plt.plot(test_data)
        # plt.scatter(anomalies, test_data[anomalies], color='r', label="True Anomalies")
        # plt.scatter(detected_anomalies, test_data[detected_anomalies], color='g', label="Detected Anomalies")
        
        plt.xlabel("Time")
        plt.ylabel("Value")

        # Create custom legend handles
        true_anomalies_handle = mlines.Line2D([], [], color='r', linewidth=2, label='True Anomalies')
        detected_anomalies_handle = mlines.Line2D([], [], color='g', linewidth=2, label='Detected Anomalies')
        data_handle = mlines.Line2D([], [], color='b', linewidth=2.5, label='Data')

        # Add the custom handles to the legend
        plt.legend(handles=[true_anomalies_handle, detected_anomalies_handle, data_handle])

        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(path)
        elif getfig:
            return fig
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
        self.batch_run = True  # Use this for selecting memory characteristics
        self.first_sample = True    # Use this for first sample not in batch
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
        if self.batch_run or self.first_sample:
            # Initialize hidden state with zeros
            if len(x.shape) > 2:    # If the input is a batch
                h = [torch.zeros(1, x.size(0), hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
                c = [torch.zeros(1, x.size(0), hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
            else:   # If the input is a single sample
                h = [torch.zeros(1, hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]
                c = [torch.zeros(1, hidden_size).to(device=x.device) for hidden_size in self.lstm_hidden]

            h = [torch.nn.init.xavier_normal_(h_) for h_ in h]
            c = [torch.nn.init.xavier_normal_(c_) for c_ in c]
            self.first_sample = False
        else:
            h = self.h
            c = self.c

        #for model, h_, c_ in zip(self.models, h, c):
        for i, model in enumerate(self.models):
            if isinstance(model, nn.LSTM):
                #x, (h_,c_) = model(x,(h_,c_))    # Get the output and hidden states
                x, (h[i],c[i]) = model(x,(h[i],c[i]))
            else:
                x = model(x)
        
        self.h = h
        self.c = c
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
        self.hidden_state = "batch_reset"
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
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()/len(data[0])}", flush=True)

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

    def plot_loss(self, path:str=None, getfig:bool=False):
        """
        Function to plot the loss of the model. If path is specified, the plot is saved to the path.
        """
        fig = plt.figure()
        self.criterion.loss_tracker.plot_losses()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(path)
        elif getfig:
            return fig
        else:
            plt.show()
    
    def sample_run(self):
        """
        Function to run the model on a single sample.
        """
        self.batch_run = False
        self.first_sample = True

    def batch_run(self):
        """
        Function to run the model on a batch.
        """
        self.batch_run = True

        
class VAE_LSTM(nn.Module):
    def __init__(self, vae:VAE=None, lstm:LSTM=None, input_dims:int = None, latent_dims:int = None, n_channels:int = 1, n_kernels_vae:int = 512, n_neurons_lstm:int = 64, optimizer_vae:optim.Optimizer = None, optimizer_lstm:optim.Optimizer = None, criterion_vae:nn.Module = None, criterion_lstm:nn.Module = None, device:torch.device = None, normalized:bool = True):
        
        super(VAE_LSTM, self).__init__()
        if vae is not None:
            self.vae = vae
        else:
            self.vae = VAE(input_dims, latent_dims, n_channels, n_kernels_vae, optimizer_vae, criterion_vae, device, normalized)
        self.input_dims = self.vae.input_dims
        self.latent_dims = self.vae.latent_dims
        self.device = self.vae.device

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

    def get_reconstruction_error(self, test_data:np.ndarray, device:torch.device=None):
        """
        Method to evaluate the model on a dataset.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        from Utils.processing import window_data
        BATCH_SIZE = 512

        if device is not None:
            self.device = device

        self.to(self.device)
        self.eval()
        self.lstm.sample_run() # Set the LSTM to run on a single sample with ongoing memory

        test_windowed_data = window_data(test_data, window_size=self.input_dims, stride=1).astype(np.float32)
        if len(test_windowed_data.shape) == 2:
            test_windowed_data = test_windowed_data[:, np.newaxis, :]   # Add dim for channel
        shape = test_windowed_data.shape
        pad = BATCH_SIZE - shape[0] % BATCH_SIZE
        test_windowed_data = np.pad(test_windowed_data, ((0, pad), (0, 0), (0, 0)))
        test_windowed_data = test_windowed_data.reshape(-1, BATCH_SIZE, shape[1], shape[2])

        test_windowed_data = torch.tensor(test_windowed_data).to(self.device)
        window_errors = torch.zeros(shape[0]).to(self.device)
        # with torch.no_grad():
        #     vae_outputs = self.vae(test_windowed_data)
        #     lstm_outputs = self.lstm(vae_outputs[2])
        #     outputs = self.vae.decoder(lstm_outputs)
        #     window_errors = torch.mean((test_windowed_data - outputs)**2, dim=(1, 2, 3))
        
        with torch.no_grad():
            for i, input in enumerate(test_windowed_data):
                if i == len(test_windowed_data) - 1:    # If the last batch
                    input = input[:shape[0] % BATCH_SIZE]
                vae_output = self.vae(input)
                lstm_output = self.lstm(vae_output[2]) # Using on going memory versus batch memory
                output = self.vae.decoder(lstm_output)
                window_errors[i*BATCH_SIZE:i*BATCH_SIZE + len(input)] = torch.mean((input - output)**2, dim=(1, 2))
        
        reconstruction_error = torch.zeros(len(test_data)).to(self.device)
        weights = torch.zeros(len(test_data)).to(self.device)
        for i in range(len(window_errors)):
            weights[i:i+self.input_dims] += 1
            reconstruction_error[i:i+self.input_dims] += window_errors[i]
        reconstruction_error /= weights # Average the reconstruction error over the windows
        return reconstruction_error
         
    def loc_anomalies(self, reconstruction_error:torch.Tensor, threshold:float=0.9, threshold_option:int=1):
        """
        Method to locate anomalies in the reconstruction error.
        Args:
            reconstruction_error (torch.Tensor): Reconstruction error from the model.
            threshold (float): Threshold for anomaly detection.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
        """
        #threshold = torch.tensor(threshold).to(reconstruction_error.device)
        if threshold_option == 1:
            # Option 1: Use the mean and standard deviation of the reconstruction error
            error_threshold = torch.mean(reconstruction_error) + threshold * torch.std(reconstruction_error)
            return torch.where(reconstruction_error > error_threshold)[0].to('cpu')
        elif threshold_option == 2:
            # Option 2: Use percentage between max and min
            error_threshold = threshold * (torch.max(reconstruction_error) - torch.min(reconstruction_error)) + torch.min(reconstruction_error)
            return torch.where(reconstruction_error > error_threshold)[0].to('cpu')
        else:
            raise ValueError("Threshold option must be 1 or 2.")

    def get_anomalies(self, test_data:np.ndarray, threshold:float=0.9, threshold_option:int=1, device:torch.device=None):
        """
        Method to get error from the model on a dataset.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
                If None, return the reconstruction error.
                If not None, return the indices of the anomalies.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        reconstruction_error = self.get_reconstruction_error(test_data, device)
        if threshold is None:
            return reconstruction_error
        else:
            return self.loc_anomalies(reconstruction_error, threshold, threshold_option)
        
    def get_anomaly_bool(self, test_data:np.ndarray, threshold:float=0.9, threshold_option:int=1, device:torch.device=None):
        """
        Method to get anomaly locations as a boolean array.
        Shares the same shape as the input data.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
                Default is 0.9.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
        """
        anomaly_indices = self.get_anomalies(test_data, threshold, threshold_option, device)
        anomaly_bool = np.zeros(len(test_data), dtype=bool)
        anomaly_bool[anomaly_indices] = True
        return anomaly_bool
    
    def score_anomalies(self, detected_anomalies:torch.Tensor, true_anomalies:np.ndarray):
        """
        Get the precision, recall, and F1 score for the detected anomalies.
        Args:
            detected_anomalies (torch.Tensor): Anomalies detected by the model.
            true_anomalies (np.ndarray): True anomalies in the data.
        """
        true_positives = len(np.intersect1d(detected_anomalies, true_anomalies))
        false_positives = len(np.setdiff1d(detected_anomalies, true_anomalies))
        false_negatives = len(np.setdiff1d(true_anomalies, detected_anomalies))
        
        if true_positives + false_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        
        recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_score
    
    def augment_anomalies(self, detected_anomalies:torch.Tensor, true_anomalies:np.ndarray):
        """
        Method for reducing impact of consecutive windows with the same anomaly.
        Both detected anomalies and true anomalies are lists of indices.
        Args:
            detected_anomalies (torch.Tensor): Anomalies detected by the model.
            true_anomalies (np.ndarray): True anomalies in the data.
        Returns:
            torch.Tensor: Detected anomalies after discounting.
        """
        # extended_detected_anomalies = list(detected_anomalies)
        # for tru_anom in true_anomalies:
        #     for det_anom in detected_anomalies:
        #         if det_anom in tru_anom:
        #             original_det = set(extended_detected_anomalies)
        #             current_anom = set([tru_anom])
        #             extended_detected_anomalies += list(original_det - current_anom)

        extended_detected_anomalies = list(detected_anomalies)
        extended_detected_anomalies_set = set(extended_detected_anomalies)
        
        for true_anomaly in true_anomalies:
            if true_anomaly not in extended_detected_anomalies_set:
                extended_detected_anomalies.append(true_anomaly)
        
        extended_detected_anomalies.sort()
        
        return torch.tensor(extended_detected_anomalies).sort().values

        
    def evaluate(self, test_data:np.ndarray, anomalies:np.ndarray, threshold_option:int=1, device:torch.device=None, verbose:bool=True, plot:bool=False, path:str=None, getfig:bool=False):
        """
        Method to evaluate the model on a dataset. Requires the true anomalies.
        Attempts to find the best threshold for anomaly detection.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            anomalies (np.ndarray): True anomalies in the data.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
            verbose (bool): Whether to print the results.
                Default is True.
            plot (bool): Whether to plot the results.
                Default is False.
            path (str): Path to save the plot.
                Default is None.
            getfig (bool): Whether to return the figure.
        Returns:
            float: Best threshold for anomaly detection.
            float: Precision at the best threshold.
            float: Recall at the best threshold.
            float: F1 Score at the best threshold.
            float: Best threshold for augmented anomaly detection.
            float: Precision at the best threshold for augmented anomaly detection.
            float: Recall at the best threshold for augmented anomaly detection.
            float: F1 Score at the best threshold for augmented anomaly detection.
            fig: Figure object if getfig is True.
        """
        reconstruction_error = self.get_reconstruction_error(test_data, device)
        if anomalies is None:
            raise ValueError("Anomalies must be specified.")
        
        anomalies = np.array(anomalies)
        test_data = np.array(test_data)

        if isinstance(anomalies, np.ndarray) and anomalies.dtype == bool and len(anomalies) == len(test_data):
            anom_loc = np.where(anomalies)[0]
        elif isinstance(anomalies, np.ndarray) and anomalies.dtype != np.int64 and len(anomalies) < len(test_data):
            anom_loc = anomalies
        else:
            raise ValueError("Anomalies must be a list or numpy array of indices or booleans.")
        anom_loc = torch.tensor(anom_loc)
        
        # threshold_list = np.linspace(0.1, 1, 10)
        # threshold_list = np.linspace(0.4, 1, 25)
        threshold_list = np.concatenate((np.linspace(0.1, 1, 37), np.linspace(1.5, 10, 18)))

        precisions = np.zeros(len(threshold_list))
        recalls = np.zeros(len(threshold_list))
        f1_scores = np.zeros(len(threshold_list))
        aug_precisions = np.zeros(len(threshold_list))
        aug_recalls = np.zeros(len(threshold_list))
        aug_f1_scores = np.zeros(len(threshold_list))
        for i, threshold in enumerate(threshold_list):
            detected_anomalies = self.loc_anomalies(reconstruction_error, threshold, threshold_option)
            precisions[i], recalls[i], f1_scores[i] = self.score_anomalies(detected_anomalies, anom_loc)
            aug_detected_anomalies = self.augment_anomalies(detected_anomalies, anom_loc)
            aug_precisions[i], aug_recalls[i], aug_f1_scores[i] = self.score_anomalies(aug_detected_anomalies, anom_loc)
            if verbose:
                print(f"Threshold: {threshold}, Precision: {precisions[i]}, Recall: {recalls[i]}, F1 Score: {f1_scores[i]}")
                print(f"\tAugmented Precision: {aug_precisions[i]}, Augmented Recall: {aug_recalls[i]}, Augmented F1 Score: {aug_f1_scores[i]}")

        best_index = np.argmax(f1_scores)
        best_index_aug = np.argmax(aug_f1_scores)

        if plot or path is not None or getfig:  
            fig = plt.figure(figsize=(12, 6))

            # Plotting the first subplot
            plt.subplot(1, 2, 1)
            plt.plot(threshold_list, precisions, label="Precision")
            plt.plot(threshold_list, recalls, label="Recall")
            plt.plot(threshold_list, f1_scores, label="F1 Score")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()

            # Plotting the second subplot
            plt.subplot(1, 2, 2)
            plt.plot(threshold_list, aug_precisions, label="Precision")
            plt.plot(threshold_list, aug_recalls, label="Recall")
            plt.plot(threshold_list, aug_f1_scores, label="F1 Score")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()

            if path is not None:
                folder = os.path.dirname(path)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                plt.savefig(path)
            elif getfig:
                return threshold_list[best_index], precisions[best_index], recalls[best_index], f1_scores[best_index], threshold_list[best_index_aug], aug_precisions[best_index_aug], aug_recalls[best_index_aug], aug_f1_scores[best_index_aug], fig
            else:
                plt.show()

        return threshold_list[best_index], precisions[best_index], recalls[best_index], f1_scores[best_index], threshold_list[best_index_aug], aug_precisions[best_index_aug], aug_recalls[best_index_aug], aug_f1_scores[best_index_aug]

    def plot_anomaly(self, test_data:np.ndarray, threshold, anomalies:np.ndarray=None, Fs=1, normalize:bool=True, threshold_option:int=1, device:torch.device=None, path:str=None, getfig:bool=False):
        """
        Method to plot the anomalies detected by the model.
        Args:
            test_data (np.ndarray): Test data to be evaluated.
            threshold (float): Threshold for anomaly detection.
            anomalies (np.ndarray): True anomalies in the data. Boolean array.
            Fs (int): Sampling frequency of the data.
            normalize (bool): Whether to normalize the data before plotting.
            threshold_option (int): Option for threshold calculation.
                1: Use the mean and standard deviation of the reconstruction error.
                2: Use percentage between max and min.
                Default is 1.
            device (torch.device): Device to be used for training.
                Default is None.
            path (str): Path to save the plot.
                Default is None.
            getfig (bool): Whether to return the figure.
        """
        test_data = np.array(test_data)

        if device is not None:
            self.device = device
        if normalize:
            show_data = test_data.copy()
            test_data = (test_data - np.mean(test_data)) / np.std(test_data)

        if anomalies is not None:
            anomalies = np.array(anomalies)
            if isinstance(anomalies, np.ndarray) and anomalies.dtype == bool and len(anomalies) == len(test_data):
                anom_loc = np.where(anomalies)[0]
            elif isinstance(anomalies, np.ndarray) and anomalies.dtype != np.int64 and len(anomalies) < len(test_data):
                anom_loc = anomalies
            else:
                raise ValueError("Anomalies must be a list or numpy array of indices or booleans.")
            anom_loc = torch.tensor(anom_loc)

        #detected_anomalies = self.get_anomaly_bool(test_data, threshold=threshold, threshold_option=threshold_option, device=self.device)
        detected_anomalies = self.get_anomalies(test_data, threshold=threshold, threshold_option=threshold_option, device=self.device)

        t = np.arange(len(test_data)) / Fs
        fig = plt.figure()
        # plt.plot(t, test_data)

        #plt.vlines(t[detected_anomalies], ymin=np.min(test_data), ymax=np.max(test_data), colors='g', linestyles='solid', label="Detected Anomalies", alpha=0.5, linewidth=1)

        if anomalies is not None:
           plt.vlines(t[anom_loc], ymin=np.min(show_data), ymax=np.max(show_data), colors='r', linestyles='solid', label="True Anomalies", alpha=0.5, linewidth=0.05)
        plt.vlines(t[detected_anomalies], ymin=np.min(show_data), ymax=np.max(show_data), colors='g', linestyles='solid', label="Detected Anomalies", alpha=0.5, linewidth=0.1)

        if normalize:
            test_data = show_data

        plt.plot(t, test_data, linewidth=2.5, color='black')
        plt.plot(t, test_data, label="Data", color='b')

        
        # plt.axvline(x=t[detected_anomalies[0]], color='g', linestyle='solid', label="Detected Anomalies")
        # for detected_anom in detected_anomalies[1:]:
        #     plt.axvline(x=t[detected_anom], color='g', linestyle='solid')

        # if anomalies is not None:
        #     #plt.scatter(t[anomalies], test_data[anomalies], color='r', label="True Anomalies")
        #     plt.axvline(x=t[anom_loc[0]], color='r', linestyle='--', label="True Anomalies") 
        #     for anom in anom_loc[1:]:
        #         plt.axvline(x=t[anom], color='r', linestyle='--')        #plt.plot(t, detected_anomalies, label="Detected Anomalies", color='g--')
        
        # plt.plot(test_data)
        # plt.scatter(anomalies, test_data[anomalies], color='r', label="True Anomalies")
        # plt.scatter(detected_anomalies, test_data[detected_anomalies], color='g', label="Detected Anomalies")
        
        plt.xlabel("Time")
        plt.ylabel("Value")

        # Create custom legend handles
        true_anomalies_handle = mlines.Line2D([], [], color='r', linewidth=2, label='True Anomalies')
        detected_anomalies_handle = mlines.Line2D([], [], color='g', linewidth=2, label='Detected Anomalies')
        data_handle = mlines.Line2D([], [], color='b', linewidth=2.5, label='Data')

        # Add the custom handles to the legend
        plt.legend(handles=[true_anomalies_handle, detected_anomalies_handle, data_handle])

        if path is not None:
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(path)
        elif getfig:
            return fig
        else:
            plt.show()