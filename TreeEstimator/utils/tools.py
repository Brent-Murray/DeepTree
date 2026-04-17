import glob
import os
from datetime import datetime
from itertools import cycle, islice
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Dataset

from .pointcloud_metrics import calculate_metrics


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return attributes


class PointCloudsInDF(Dataset):
    def __init__(self, filepath, df):
        self.filepath = filepath
        self.df = df
        super().__init__()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        df_idx = self.df.iloc[idx]
        filename = df_idx["filename"]
        file = os.path.join(self.filepath, filename)

        # Get target label
        target = df_idx["class"]
        target = torch.tensor(target, dtype=torch.long)

        # Read in entier las file
        las = read_las(file, get_attributes=True)
        metrics = calculate_metrics(las)
        metrics = torch.tensor(list(metrics.values())).float()
        coords = np.vstack((las["X"], las["Y"], las["Z"])).transpose()
        coords = torch.from_numpy(coords).float()
        return coords, metrics, target, filename


def _init_(model_name):
    # Create folder structure
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + model_name):
        os.makedirs("checkpoints/" + model_name)
    if not os.path.exists("checkpoints/" + model_name + "/models"):
        os.makedirs("checkpoints/" + model_name + "/models")
    if not os.path.exists("checkpoints/" + model_name + "/output"):
        os.makedirs("checkpoints/" + model_name + "/output")
    if not os.path.exists("checkpoints/" + model_name + "/output/gradients"):
        os.makedirs("checkpoints/" + model_name + "/output/gradients")


def make_confusion_matrix(
    cm,
    labels,
    normalize=False,
    accuracy=None,
    precision=None,
    recall=None,
    f1=None,
    figsize=None,
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = ""

    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    vmin = np.min(cm)
    vmax = np.max(cm)
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=~off_diag_mask,
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="black",
        xticklabels=labels,
        yticklabels=labels,
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=off_diag_mask,
        cmap="Reds",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )

    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label" + stats_text)


def collate_fn(batch):
    # Unpack the batch into individual lists

    coords_list, metrics, targets, filenames = zip(*batch)

    # Determine the maximum number of points in the batch
    max_points = max(coords.size(0) for coords in coords_list)

    padded_coords = []
    for coords in coords_list:
        num_points = coords.size(0)
        # If necessary, pad with zeros along the point dimension
        if num_points < max_points:
            pad = torch.zeros(
                (max_points - num_points, coords.size(1)), dtype=coords.dtype
            )
            coords = torch.cat([coords, pad], dim=0)
        padded_coords.append(coords)

    # Stack the padded coordinates and targets into tensors.
    batch_coords = torch.stack(padded_coords)  # Shape: (batch_size, max_points, 3)
    batch_targets = torch.stack(targets)
    batch_metrics = torch.stack(metrics)

    return batch_coords, batch_metrics, batch_targets, filenames


def plot_gradients(model, save_path="gradients.png"):
    """
    Plots and saves the gradient norms for each layer of the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        save_path (str): File path to save the gradient plot.
    """
    # Prepare data for plotting
    grad_norms = []
    layer_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm < 1e-4:
                color = "\033[94m"
                print(
                    f"{color}Vanishing gradient detected in {name}: {grad_norm:.4e}\033[0m"
                )
            elif grad_norm > 1e2:
                color = "\033[91m"
                print(
                    f"{color}Exploding gradient detected in {name}: {grad_norm:.4e}\033[0m"
                )
            # else:
            #     color = "\033[32m"
            #     print(f"{color}Gradient is within range for {name}: {grad_norm:.4e}\033[0m")
            layer_names.append(name)
        elif param.requires_grad and param.grad is None:
            grad_norms.append(0.0)  # For parameters with no gradient computed
            layer_names.append(name)
            print(f"\033[93mGradient is None for: {name}\033[0m")

    # Plot gradients
    plt.figure(figsize=(12, 6))
    plt.bar(layer_names, grad_norms, alpha=0.7)
    plt.xlabel("Layers")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms by Layer")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory


def plot_loss(df, title="Training and Validation Loss", save_path="loss.png"):
    """
    Plots training and validation loss over epochs, with a red vertical line marking
    the epoch with the lowest validation loss and a horizontal line marking its value on the y-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'epoch', 'train_loss', and 'val_loss' columns.
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the plot to the specified path.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))

    # Plot training and validation loss
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="s")

    # Identify the epoch with the lowest validation loss
    min_val_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    min_val_loss = df["val_loss"].min()

    # Vertical line at the epoch with lowest validation loss
    plt.axvline(
        x=min_val_loss_epoch,
        color="red",
        linestyle="--",
        label=f"Lowest Val Loss (Epoch {min_val_loss_epoch})",
    )

    # Horizontal line from the y-axis to the validation loss point
    plt.axhline(
        y=min_val_loss,
        xmin=0,
        xmax=min_val_loss_epoch / df["epoch"].max(),
        color="red",
        linestyle="--",
    )

    # Add the loss value as an annotation outside the y-axis border
    plt.annotate(
        f"{min_val_loss:.4f}",
        xy=(plt.xlim()[0], min_val_loss),
        xytext=(-3, 0),  # Offset to move text outside the y-axis
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=12,
        color="red",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class AdaptiveLRScheduler:
    def __init__(
        self,
        optimizer,
        patience,
        factor,
        min_lr,
        switch_threshold=1.2,
        T_0=10,
        T_mult=2,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            patience: ReduceLROnPlateau patience
            factor: Multiplicative factor for ReduceLROnPlateau
            min_lr: Absolute lowest LR (eta_min for CosineAnnealing)
            switch_threshold: Factor above min_lr where we switch to cosine (prevents LR flatlining)
            T_0: First cycle duration for CosineAnnealingWarmRestarts
            T_mult: Multiplier for cycle length increase
        """
        self.optimizer = optimizer
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor,
            min_lr=min_lr * switch_threshold,
        )
        self.cosine_scheduler = None
        self.using_cosine = False
        self.min_lr = min_lr
        self.switch_threshold = switch_threshold  # Ensure LR switch is above min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.current_epoch = 0

    def step(self, metric):
        if not self.using_cosine:
            prev_lr = self.optimizer.param_groups[0]["lr"]
            self.plateau_scheduler.step(metric)
            new_lr = self.optimizer.param_groups[0]["lr"]

            # Switch slightly before hitting min_lr to allow oscillation
            if new_lr <= self.min_lr * self.switch_threshold and prev_lr != new_lr:
                print(f"Switching to CosineAnnealingWarmRestarts at LR={new_lr}")
                self.using_cosine = True

                # Ensure eta_min is lower than new_lr to enable oscillations
                eta_min = self.min_lr * 0.8  # Set eta_min slightly below min_lr

                self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=eta_min
                )

        if self.using_cosine:
            self.cosine_scheduler.step(self.current_epoch)
            self.current_epoch += 1
