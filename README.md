# Unsupervised-Anomaly-Detection
Project for detecting anomalies without ground truth.

### To do:
* Elbo Loss function (implemented, but check eval)
    * Stop using. Trying to get KL and MSE working well.
    * BCE wont work unless min max constrains input and output to 0 and 1.
* Create evaluation methods (most important...)
    * Show where anomaly was predicted.
    * With list of anomaly locations, run evaluate.
    * Need to decide how to organize get_anomalies and evaluate
* Look into multichannel
    * Added param for VAE. Just need to make sure dataloader will work correctly.