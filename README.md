# Unsupervised-Anomaly-Detection
Project for detecting anomalies without ground truth.

## Requirements
* Numpy
* PyTorch
* Matplotlib
* Sklearn

## Usage
To get started, `train_model.ipynb` is an example notebook on how to use the anomaly detection. It holds method to combine data for data loading, setting up and training models, and combining models for anomaly detection. There are many options to evaluation, and metrics and plots are provided. Use more than 10 epochs...

## Tuning
All tuning can be done by giving parameters to already implemented functions. For example, training a model will take parameters such as optimizer, criterion, ect.

* Window size for the windowed data. Example uses 100, but this is very small. For fast sampling, this should be largely increased.
* Change number of epochs. Examples has 10, but try 20, 50, 100
* Latent space for the VAE is default at 6, but a larger latent space will probably improve performance.
* VAE evaluation has not been tested, but it is identical to VAE-LSTM evaluation.
* When evaluating, there are two methods for thresholding and two different methods for getting f1 score. Why not try both?
* Currently, only MSE is working correctly for criterion (loss function), but this could be a large impact on training.
* Change batch size will greatly impact the LSTM training.
* Optimizer type, learning rate, and other parameters for optimizer.
* Number of n_kernels and n_neurons can be adjusted for the VAE and LSTM.

## Results
A paper was written for this project, formatted using latex, and stored within a folder as a pdf in this repository.

## Acknowledgements
This project was completed for EEC 289A at UC Davis, Spring 2024, by Randall Fowler, Conor King, and Ajay Suresh.

## Continuation

### To do:
* Loss Functions
    * KLD produces negative number.
    * BCE will work if input and output are scaled between 0 and 1.
    * With KLD and BCE, different combinations of summing loss functions can be done.
* Improve evaluation plot. Colors are not pleasing.
* Adjust LSTM dataset as it is training on current data rather than future data.
* Maybe train the LSTM as the entire VAE-LSTM with the VAE frozen.
* Multichannel input
    * Mostly implemented, but have not tested. Expected to have bugs. Issues will probably be in data loading and evaluation.
