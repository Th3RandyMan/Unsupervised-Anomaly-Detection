import sys
import os

cwd = os.getcwd()
cwd = os.path.dirname(cwd)
print(cwd)

from matplotlib import pyplot as plt
sys.path.append(cwd)

import glob
from typing import List
import numpy as np
import pandas as pd
from Utils.dataloader import DataLoaderGenerator
from Utils.models import LSTM, VAE, VAE_LSTM
from Utils.processing import window_data
from Utils.mdreport import MarkdownReport
from tqdm import tqdm

# Constant Training parameters
FS = 100
TRAINING_PATH = os.path.join(cwd, "TFO Data", "PPG", "Clean PPG")
TESTING_PATH = os.path.join(cwd, "TFO Data", "PPG", "Artificial Anomalies","datafiles")
EVAL_PATH = os.path.join(cwd, "TFO Data", "PPG", "Artificial Anomalies","ahomaly indices")
# BASE_NAME = "IMU_Training"
THRESHOLD_METHOD = 1

# Sweep parameters
N_EPOCHS = [20, 50, 100]
WINDOW_SIZES = [100, 200, 500]
BATCH_SIZES = [32, 128, 512]
LATENT_DIMS = [6, 12, 24]


if __name__ == "__main__":
    train_file_list = glob.glob(TRAINING_PATH + "/*.csv")
    test_file_list = glob.glob(TESTING_PATH + "/*.csv")
    eval_file_list = glob.glob(TESTING_PATH + "/*.csv")

    # window_size = 100
    latent_dim = 6
    batch_size = 32
    # n_epochs = 10
    run_name = "PPG1"

    # Create directory if it doesn't exist
    if not os.path.exists("Results/Scaling"):
        os.makedirs("Results/Scaling")
    if not os.path.exists("Results/Reports"):
        os.makedirs("Results/Reports")
    if not os.path.exists("Results/Models"):
        os.makedirs("Results/Models")

    i = 0
    with tqdm(total = len(N_EPOCHS) * len(WINDOW_SIZES) * len(BATCH_SIZES) * len(LATENT_DIMS)) as pbar:
        for n_epochs in N_EPOCHS:
            for window_size in WINDOW_SIZES:
                if window_size != 100 or n_epochs != 100:
                    i += 1
                    for p in range(len(BATCH_SIZES) * len(LATENT_DIMS)):
                        pbar.update(1)
                    continue
                # Create lists to store scaling parameters and windowed data
                mean_scaling = []
                variance_scaling = []
                windowed_data_list: List[np.ndarray] = []

                # Load data from files
                for f, file in enumerate(train_file_list):
                    # Add filter condition here if needed
                    data = pd.read_csv(file, usecols=[0])
                    data = data.to_numpy().squeeze()

                    # Normalize data
                    mean_scaling.append(np.mean(data, axis=0))
                    variance_scaling.append(np.var(data, axis=0))
                    data = (data - mean_scaling[f]) / variance_scaling[f]

                    # Window data
                    windowed_data_list.append(window_data(data, window_size))

                # Save scaling parameters                        
                epoch_win_name = run_name + f"_{n_epochs}_{window_size}"
                np.save("Results/Scaling/" + epoch_win_name + "_mean.npy", mean_scaling)
                np.save("Results/Scaling/" + epoch_win_name + "_variance.npy", variance_scaling)

                # # Create Report
                # report = MarkdownReport(f"Results/Reports/{run_name}", run_name + f"_{i}.md", title="Training Sweep Report")
                # report.add_code_report("Report Parameters", f"\
                #                         Epochs: {n_epochs}\n\
                #                         Window Size: {window_size}\n\
                #                         Training Data: {TRAINING_PATH}\n\
                #                         Testing Data: {TESTING_PATH}\n\
                #                         Threshold Method: {THRESHOLD_METHOD}\n\
                #                         ")
                i += 1
                j = 0
                for batch_size in BATCH_SIZES:
                    # Create DataLoader objects
                    data_loader_generator = DataLoaderGenerator(data=windowed_data_list, batch_size=batch_size)
                    train_loader = data_loader_generator.generate()

                    for latent_dim in LATENT_DIMS:
                        if i == 1 or latent_dim != 6:   # Skip over first buch of runs
                            j += 1
                            pbar.update(1)
                            continue
                                        # Create Report
                        report = MarkdownReport(f"Results/Reports/{run_name}", run_name + f"_{i}_{j}.md", title="Training Sweep Report")
                        report.add_code_report("Report Parameters", f"\
                                                Epochs: {n_epochs}\n\
                                                Window Size: {window_size}\n\
                                                Training Data: {TRAINING_PATH}\n\
                                                Testing Data: {TESTING_PATH}\n\
                                                Threshold Method: {THRESHOLD_METHOD}\n\
                                                ")
                        # Save params for report
                        report.add_text_report(f"Model Training Runs", f"Results of Training Run {j}:\n")
                        report.add_code_report("Model Parameters", f"\
                                                Latent Dim: {latent_dim}\n\
                                                Batch Size: {batch_size}\n\
                                                ")

                        # Create and train model
                        vae = VAE(window_size, latent_dim)
                        vae.train_model(train_loader, n_epochs=n_epochs)
                        #vae.plot_loss("Results/Loss/"+ run_name + "_vae_loss.png")
                        vae_loss_fig = vae.plot_loss(getfig=True)

                        # Save model
                        vae.save_model("Results/Models/" + run_name + f"_{n_epochs}_{window_size}_{batch_size}_{latent_dim}" + "_vae_model.pth")

                        # Create lstm data
                        encoded_training_data = vae.encode_data(train_loader)
                        lstm_dataloader = DataLoaderGenerator(encoded_training_data, batch_size=batch_size)
                        lstm_train_loader = lstm_dataloader.generate()

                        # Create and train lstm model
                        lstm = LSTM(latent_dim)
                        lstm.train_model(lstm_train_loader, n_epochs=n_epochs)
                        #lstm.plot_loss("Results/Loss/" + run_name + "_lstm_loss.png")
                        lstm_loss_fig = lstm.plot_loss(getfig=True)

                        # Save lstm model
                        lstm.save_model("Results/Models/" + run_name + f"_{n_epochs}_{window_size}_{batch_size}_{latent_dim}" + "_lstm_model.pth")

                        # Save Loss figures in report
                        report.add_image_report("VAE Loss", vae_loss_fig)
                        report.add_image_report("LSTM Loss", lstm_loss_fig)

                        # Evaluate model
                        vae_lstm = VAE_LSTM(vae, lstm)
                        
                        # Load testing data
                        for file in test_file_list:
                            report.add_text_report(f"Run {j} Testing Data", f"Testing on {file}\n")
                            df = pd.read_csv(file)
                            
                            key = file.split('_')[-1].split('.')[0]
                            eval_file = [filename for filename in eval_file_list if filename.endswith(key + '.csv')][0]
                            df_eval = pd.read_csv(file)

                            channel = df.columns[0]
                            
                            eval_output = vae_lstm.evaluate(df[channel], df_eval[channel], threshold_option=THRESHOLD_METHOD, getfig=True)
                            print() # Separate between evaluation outputs
                            
                            # Save evaluation metrics
                            eval_metric_str = f"Best Threshold: {eval_output[0]}\n\
                                                Precision: {eval_output[1]}\n\
                                                Recall: {eval_output[2]}\n\
                                                F1 Score: {eval_output[3]}\n\
                                                \n\
                                                Augmented:\n\
                                                    \tBest Threshold: {eval_output[4]}\n\
                                                    \tPrecision: {eval_output[5]}\n\
                                                    \tRecall: {eval_output[6]}\n\
                                                    \tF1 Score: {eval_output[7]}\n"
                            
                            report.add_code_report("Evaluation Metrics", eval_metric_str)
                        
                            # Save evaluation figures
                            report.add_image_report("Evaluation Metrics Plot", eval_output[8])

                            # Save anomaly detection plots
                            anom_fig = vae_lstm.plot_anomaly(df['Dirty'], eval_output[0], df['Anomaly'], getfig=True)
                            anom_aug_fig = vae_lstm.plot_anomaly(df['Dirty'], eval_output[4], df['Anomaly'], getfig=True)
                            report.add_image_report("Anomaly Detection Plot", anom_fig)
                            report.add_image_report("Anomaly Detection Plot (Augmented)", anom_aug_fig)

                            # Close figures
                            plt.close('all')    # Maybe close each figure individually
                        report.save_report()
                        j += 1
                        pbar.update(1)
                        for p in range(3):
                            print()
                # Save report
                #report.save_report()






