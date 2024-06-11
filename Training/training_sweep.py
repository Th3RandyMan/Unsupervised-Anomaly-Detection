import os
# # Change current working directory to parent directory
# os.chdir("..")

import glob
from typing import List
import numpy as np
import pandas as pd
from Utils.dataloader import DataLoaderGenerator
from Utils.models import LSTM, VAE
from Utils.processing import window_data


# Constant Training parameters
FS = 100
TRAINING_PATH = r"C:\Users\randa\OneDrive - University of California, Davis\Documents\Davis\2024 xSpring\EEC 289A\Unsupervised-Anomaly-Detection\TFO Data\Clean IMU"















if __name__ == "__main__":
    file_list = glob.glob(TRAINING_PATH + "/*.csv")

    window_size = 100
    latent_dim = 6
    batch_size = 32
    n_epochs = 10
    run_name = "test_run"

    mean_scaling = []
    variance_scaling = []
    windowed_data_list: List[np.ndarray] = []

    # Load data from files
    for i, file in enumerate(file_list):
        # Add filter condition here if needed
        data = pd.read_csv(file)
        data = data.to_numpy()

        # Normalize data
        mean_scaling.append(np.mean(data, axis=0))
        variance_scaling.append(np.var(data, axis=0))
        data = (data - mean_scaling[i]) / variance_scaling[i]

        # Window data
        windowed_data_list.append(window_data(data, window_size))

    # Save scaling parameters
    np.save("Results/Scaling/" + run_name + "_mean_scaling.npy", mean_scaling)
    np.save("Results/Scaling/" + run_name + "_variance_scaling.npy", variance_scaling)
    
    # Create DataLoader objects
    data_loader_generator = DataLoaderGenerator(data=windowed_data_list, batch_size=batch_size)
    train_loader = data_loader_generator.generate()

    # Create and train model
    vae = VAE(input_dim=window_size, latent_dim=latent_dim)
    vae.train_model(train_loader, n_epochs=n_epochs)
    vae.plot_loss("Results/Loss/"+ run_name + "_vae_loss.png")

    # Save model
    vae.save_model("Results/Models/" + run_name + "_vae_model.pth")

    # Create lstm data
    encoded_training_data = vae.encode_data(train_loader)
    lstm_dataloader = DataLoaderGenerator(encoded_training_data, batch_size=batch_size)
    lstm_train_loader = lstm_dataloader.generate()

    # Create and train lstm model
    lstm = LSTM(input_dim=latent_dim)
    lstm.train_model(lstm_train_loader, n_epochs=n_epochs)
    lstm.plot_loss("Results/Loss/" + run_name + "_lstm_loss.png")

    # Save lstm model
    lstm.save_model("Results/Models/" + run_name + "_lstm_model.pth")





