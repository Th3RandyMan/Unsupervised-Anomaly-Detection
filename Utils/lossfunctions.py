"""
A set of custom loss function meant to be used with the ModelTrainer Class
created by Rishad Joarder.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from matplotlib import pyplot as plt
import torch


DATA_LOADER_INPUT_INDEX, DATA_LOADER_LABEL_INDEX, DATA_LOADER_EXTRA_INDEX = 0, 1, 2


class LossTracker:
    """
    A class to track the loss values during training. This class is meant to be used with the ModelTrainer class
    """

    def __init__(self, loss_names: List[str]):
        self.tracked_losses = loss_names
        # Set types for the dictionaries
        self.epoch_losses: Dict[str, List[float]]
        # Create the dictionaries
        self.epoch_losses = {loss_name: [] for loss_name in loss_names}  # Tracks the average loss for each epoch
        self.step_loss_sum = {loss_name: 0.0 for loss_name in loss_names}  # Tracks the loss for each step(in an epoch)
        self.steps_per_epoch_count = {
            loss_name: 0 for loss_name in loss_names
        }  # Tracks the number of steps in an epoch

    def step_update(self, loss_name: str, loss_value: float) -> None:
        """
        Update the losses for a single step within an epoch by appending the loss to the per_step_losses dictionary
        """
        if loss_name in self.step_loss_sum:
            self.steps_per_epoch_count[loss_name] += 1
            self.step_loss_sum[loss_name] += loss_value
        else:
            raise ValueError(f"Loss name '{loss_name}' not found in the LossTracker list!")

    def epoch_update(self) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch.

        Averages out the losses from all the steps and places the average onto the epoch_losses list. Clears the
        per step losses for the next epoch
        """
        for loss_name in self.epoch_losses.keys():
            if self.steps_per_epoch_count[loss_name] != 0:
                self.epoch_losses[loss_name].append(
                    self.step_loss_sum[loss_name] / self.steps_per_epoch_count[loss_name]
                )
                self.step_loss_sum[loss_name] = 0.0  # Reset
                self.steps_per_epoch_count[loss_name] = 0  # Reset

    def plot_losses(self) -> None:
        """
        Plot the losses on the current axes
        """
        if len(self.epoch_losses) == 0:
            print("No losses to plot!")
            return
        for loss_name, loss_values in self.epoch_losses.items():
            plt.plot(loss_values, label=loss_name)
            plt.legend()

    def reset(self):
        """
        Clears out all saved losses
        """
        for loss_name in self.epoch_losses.keys():
            self.epoch_losses[loss_name] = []
            self.step_loss_sum[loss_name] = 0.0
            self.steps_per_epoch_count[loss_name] = 0


class LossFunction(ABC):
    """
    Base abstract class for all loss functions. All loss functions must inherit from this class and implement the
    following methods
        |1. __call__ (required) : When called, this method should return the loss value
        |2. __str__  (required) : Return a string representation of the loss function
        |3. loss_tracker_epoch_update (optional) : Called at the end of the epoch to perform any necessary operations on
        |                                          the LossTracker object. This should usually include updating the
        |                                          LossTracker object with the average loss for the epoch
        |4. reset (optional) : Reset the loss function
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.train_loss_name = "train_loss" if name is None else f"{name}_train_loss"
        self.val_loss_name = "val_loss" if name is None else f"{name}_val_loss"
        self.loss_tracker = LossTracker([self.train_loss_name, self.val_loss_name])

    @abstractmethod
    def __call__(self, model_output, dataloader_data, trainer_mode: str) -> torch.Tensor:
        """
        Calculate & return the loss
        :param model_output: The output of the model
        :param dataloader_data: The data from the dataloader (The length of this depends on the DataLoader used)
        :param trainer_mode: The mode of the trainer (train/validate)

        Implementation Notes: When implementing this method, make sure to update the loss tracker using the
        loss_tracker_step_update method
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the loss function
        """

    def loss_tracker_step_update(self, loss_value: float, trainer_mode: str) -> None:
        """
        The Defualt loss tracker step update method. This is meant to be called at the very end of the __call__ method.
        If you want some custom behavior, you can override this method in the subclass
        """
        loss_name = self.train_loss_name if trainer_mode == "train" else self.val_loss_name
        self.loss_tracker.step_update(loss_name, loss_value)

    def reset(self) -> None:
        """
        Reset
        """
        self.loss_tracker = LossTracker(list(self.loss_tracker.epoch_losses.keys()))

    def loss_tracker_epoch_update(self) -> None:
        """
        Called at the end of the epoch to perform any necessary operations in the ModelTrainer
        """
        self.loss_tracker.epoch_update()


class TorchLossWrapper(LossFunction):
    """
    A simple wrapper around torch.nn loss functions. This lets us seemlessly integrate torch loss functions with our own
    ModelTrainer class.

    By default, it trackes two losses:
        1.  train_loss
        2.  val_loss
    """

    def __init__(self, torch_loss_object, column_indices: Optional[List[int]] = None, name: Optional[str] = None):
        """
        :param torch_loss_object: An initialized torch loss object
        :param column_indices: The indices of the columns to be used for the loss calculation. If None, all columns are
        considered. Defaults to None
        """
        super().__init__(name)
        self.loss_func = torch_loss_object
        self.column_indices = column_indices

    def __call__(self, model_output, dataloader_data, trainer_mode):
        if self.column_indices is not None:
            loss = self.loss_func(
                model_output[:, self.column_indices], dataloader_data[DATA_LOADER_LABEL_INDEX][:, self.column_indices]
            )
        else:
            loss = self.loss_func(model_output, dataloader_data[DATA_LOADER_LABEL_INDEX])

        # Update internal loss tracker
        self.loss_tracker_step_update(loss.item(), trainer_mode)
        return loss

    def __str__(self) -> str:
        return f"Torch Loss Function: {self.loss_func}"


class SumLoss(LossFunction):
    """
    Sum of two loss functions

    Special note: The name of the independent losses tracked inside each LossFunction needs to unique between all losses
    being summed

    Addional notes: The internal loss does not update per step. Rather per epoch. To get per step values, you need to
    look at the loss tracker for the individual loss_funcs/i.e. the constituents
    """

    def __init__(self, loss_funcs: List[LossFunction], weights: List[float], name: Optional[str] = None):
        super().__init__(name)
        # Check validitiy of the loss names
        all_names = []
        self.loss_directory = []  # Holds a list of losses per constituent loss func.
        for loss_func in loss_funcs:
            all_names = all_names + list(loss_func.loss_tracker.epoch_losses.keys())
            self.loss_directory.append(list(loss_func.loss_tracker.epoch_losses.keys()))
        # must be unique
        assert len(all_names) == len(set(all_names)), "Loss function names should be unique!"

        # Check validity of the weights
        assert len(loss_funcs) == len(weights), "Number of loss functions and weights should match!"

        self.loss_funcs = loss_funcs
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)
        self.weights_list = weights
        self.train_losses = list(filter(lambda x: "train_loss" in x, all_names))
        self.val_losses = list(filter(lambda x: "val_loss" in x, all_names))
        # Overrides defualt loss tracker
        self.loss_tracker = LossTracker(all_names + [self.train_loss_name, self.val_loss_name])

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        # Calculate one loss to get the typing correct
        device = model_output.device if isinstance(model_output, torch.Tensor) else model_output[0].device
        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for loss_func, weight in zip(self.loss_funcs, self.weights_tensor):
            extra_loss_term = weight * loss_func(model_output, dataloader_data, trainer_mode)
            loss = torch.add(loss, extra_loss_term)

        return loss

    def __str__(self) -> str:
        individual_loss_descriptions = [str(loss_func) for loss_func in self.loss_funcs]
        individual_loss_descriptions = "\n".join(individual_loss_descriptions)
        return f"""Sum of multiple loss functions. 
        Constituent Losses: {[func.name for func in self.loss_funcs]}
        Weights: {self.weights_list}
        Individual Loss Func Description:
        {individual_loss_descriptions}
        """

    def loss_tracker_epoch_update(self) -> None:
        # Update individual loss trackers
        for loss_func in self.loss_funcs:
            loss_func.loss_tracker_epoch_update()

        # Copy the latest epoch loss from each loss function to the sum loss tracker
        for loss_func, weight in zip(self.loss_funcs, self.weights_list):
            # All losses within this tracker has the same weight
            for loss_name in loss_func.loss_tracker.epoch_losses.keys():
                # Check if there is a new loss value, if available go and copy that onto the SumLoss's loss tracker
                if len(self.loss_tracker.epoch_losses[loss_name]) < len(loss_func.loss_tracker.epoch_losses[loss_name]):
                    loss_value = loss_func.loss_tracker.epoch_losses[loss_name][-1] * weight
                    self.loss_tracker.epoch_losses[loss_name].append(loss_value)

        # Sum the losses into the train and val losses
        # Only update if the lenghts are different
        dummy_loss = self.train_losses[0]
        if len(self.loss_tracker.epoch_losses[dummy_loss]) > len(self.loss_tracker.epoch_losses[self.train_loss_name]):
            train_loss_sum = 0.0
            for loss_name in self.train_losses:
                train_loss_sum += self.loss_tracker.epoch_losses[loss_name][-1]
            self.loss_tracker.epoch_losses[self.train_loss_name].append(train_loss_sum)
        
        dummy_loss = self.val_losses[0]
        if len(self.loss_tracker.epoch_losses[dummy_loss]) > len(self.loss_tracker.epoch_losses[self.val_loss_name]):
            val_loss_sum = 0.0
            for loss_name in self.val_losses:
                val_loss_sum += self.loss_tracker.epoch_losses[loss_name][-1]
            self.loss_tracker.epoch_losses[self.val_loss_name].append(val_loss_sum)


class DynamicWeightLoss(LossFunction):
    """
    Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings
    """

    def __init__(
        self,
        loss_func: LossFunction,
        start_weight: float,
        end_weight: float,
        epoch_count: int,
        start_delay: int = 0,
        name: Optional[str] = None,
    ):
        """
        Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings.
        If start delay is set to 0, the weight will start changing from the first epoch. Otherwise, it will hold the
        start value for the first start_delay epochs before starting to change.

        Args:
            loss_func: The underlying loss function
            start_weight: The weight at the start of the epochs
            end_weight: The weight at the end of the epochs
            epoch_count: The total number of epochs
            start_delay: The number of epochs to wait before starting the weight change. During these epochs, the
            start_weight will be used
            name: The name of the loss function

        Parameters:
            current_epoch: The current epoch number. Current epoch starts at 0 and goes up to
            (total_epochs + start_delay - 1). After that point, it stops changing until reset is called
            total_epochs: The total number of epochs as defined by the user for loss weight scheduling. This does not
            include the start_delay(explained below).
            start_delay: The number of epochs to wait before starting the weight change
            weights: The weights for each epoch
            loss_tracker: The loss tracker object
        """
        super().__init__(name)
        self.loss_func = loss_func
        self.weights = torch.linspace(start_weight, end_weight, epoch_count)
        self.weights = torch.cat([start_weight * torch.ones(start_delay), self.weights])
        self.current_epoch = 0
        self.total_epochs = epoch_count
        self.start_delay = start_delay

    def __call__(self, model_output, dataloader_data, trainer_mode):
        loss = self.weights[self.current_epoch].item() * self.loss_func(model_output, dataloader_data, trainer_mode)
        # Update internal loss tracker
        self.loss_tracker_step_update(loss.item(), trainer_mode)
        return loss

    def loss_tracker_epoch_update(self) -> None:
        if self.current_epoch < self.total_epochs + self.start_delay - 1:
            self.current_epoch += 1
        self.loss_tracker.epoch_update()

    def __str__(self) -> str:
        return f"Loss Function with changing weight: {self.loss_func},\n Start Weight: {self.weights[0].item()}, End Weight: {self.weights[-1].item()}, Total Epochs: {self.total_epochs}, Start Delay: {self.start_delay}"

    def reset(self) -> None:
        self.current_epoch = 0
        self.loss_tracker = LossTracker([self.train_loss_name, self.val_loss_name])


class KLDLoss(LossFunction):
    def __init__(self, name: str = "kld"):
        super().__init__(name)
    
    def __call__(self, model_output, dataloader_data, trainer_mode: str) -> torch.Tensor:
        _, mu, logvar = model_output
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.loss_tracker_step_update(KLD.item(), trainer_mode)
        return KLD
    
    def __str__(self):
        return f"KL Divergence Loss between the latent space and the normal distribution"
