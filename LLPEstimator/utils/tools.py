import ast
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

        # Target label
        target = df_idx["perc_specs"]
        if isinstance(target, str):
            target = ast.literal_eval(target)
        target = torch.tensor(target, dtype=torch.float32)

        # Read LAS file
        las = read_las(file, get_attributes=True)
        coords = np.vstack((las["X"], las["Y"], las["Z"])).T.astype(
            np.float32
        )  # [N, 3]

        tree_ids = np.array(las["TreeID"])

        unique_ids = np.unique(tree_ids)
        subset_coords = []
        metrics_list = []

        for tid in unique_ids:
            mask = tree_ids == tid
            tree_coords = coords[mask]
            n_points = tree_coords.shape[0]

            if n_points <= 100 or n_points >= 4000:
                continue
            subset_coords.append(torch.from_numpy(tree_coords.T))

            tree_las = {k: v[mask] for k, v in las.items()}
            m = calculate_metrics(tree_las)
            metrics_list.append(torch.tensor(list(m.values()), dtype=torch.float32))

        num_trees = len(subset_coords)
        if num_trees == 0:
            # return empty outputs if no trees meet criteria
            empty_coords = torch.empty((0, 3, 0), dtype=torch.float32)
            empty_metrics = torch.empty((0,), dtype=torch.float32)
            return empty_coords, empty_metrics, {}, target

        max_points = max(c.shape[1] for c in subset_coords)
        coords_tensor = torch.zeros((num_trees, 3, max_points), dtype=torch.float32)
        for i, c in enumerate(subset_coords):
            n = c.shape[1]
            coords_tensor[i, :, :n] = c

        metrics_tensor = torch.stack(metrics_list)

        return coords_tensor, metrics_tensor, target, filename


def collate_fn(batch):
    # Unpack
    coords_list, metrics_list, targets, filenames = zip(*batch)
    B = len(batch)

    # Max sizes (handle empties)
    max_trees = max((c.shape[0] for c in coords_list), default=0)
    max_points = max((c.shape[2] for c in coords_list if c.numel() > 0), default=0)
    feature_dim = max((m.shape[1] for m in metrics_list if m.ndim == 2), default=0)

    # Allocate
    padded_coords = torch.zeros((B, max_trees, 3, max_points), dtype=torch.float32)
    padded_metrics = torch.zeros((B, max_trees, feature_dim), dtype=torch.float32)

    # Targets (assumes consistent shapes)
    target_tensor = torch.stack(targets)

    # Fill
    for i in range(B):
        # coords: (Ti, 3, Pi)
        c = coords_list[i]
        Ti = c.shape[0]
        Pi = c.shape[2] if c.numel() > 0 else 0
        if Ti > 0 and Pi > 0:
            padded_coords[i, :Ti, :, :Pi] = c

        # metrics: (Ti, M) or empty (0,)
        m = metrics_list[i]
        if m.ndim == 2 and Ti > 0 and feature_dim > 0:
            Mi = m.shape[1]
            padded_metrics[i, :Ti, :Mi] = m  # zero-fills any extra dims

    return {
        "coords": padded_coords,  # [B, max_trees, 3, max_points]
        "metrics": padded_metrics,  # [B, max_trees, feature_dim]
        "targets": target_tensor,  # [B, ...]
        "filenames": filenames,
    }


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
    if not os.path.exists("checkpoints/" + model_name + "/output/species_predictions"):
        os.makedirs("checkpoints/" + model_name + "/output/species_predictions")


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
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker="s")

    min_idx = df["val_loss"].idxmin()
    min_val_loss_epoch = float(df.at[min_idx, "epoch"])
    min_val_loss = float(df.at[min_idx, "val_loss"])

    plt.axvline(
        x=min_val_loss_epoch,
        color="red",
        linestyle="--",
        label=f"Lowest Val Loss (Epoch {int(min_val_loss_epoch)})",
    )

    xmax = float(min_val_loss_epoch) / float(df["epoch"].max())
    plt.axhline(
        y=min_val_loss,
        xmin=0,
        xmax=xmax,
        color="red",
        linestyle="--",
    )

    plt.annotate(
        f"{min_val_loss:.4f}",
        xy=(plt.xlim()[0], min_val_loss),
        xytext=(-3, 0),
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


class SingleTreeDataset(Dataset):
    """
    Each file contains ONE tree (no TreeID segmentation).
    Returns coords [1, 3, P], metrics [1, M], filename.
    """

    def __init__(self, root_dir, file_list, min_pts=100, max_pts=100000):
        self.root_dir = root_dir
        self.files = file_list
        self.min_pts = min_pts
        self.max_pts = max_pts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root_dir, fname)

        las = read_las(path, get_attributes=True)
        coords = np.vstack((las["X"], las["Y"], las["Z"])).T.astype(np.float32)  # [P,3]
        P = coords.shape[0]
        if P < self.min_pts or P > self.max_pts:
            # return an empty example; the collate will pad; we will skip empties in test loop
            coords_tensor = torch.zeros((1, 3, 0), dtype=torch.float32)
            metrics_tensor = torch.zeros((1, 0), dtype=torch.float32)
            return coords_tensor, metrics_tensor, fname

        # metrics from the whole file (single tree)
        m = calculate_metrics(las)  # dict -> consistent key order inside impl
        metrics_vec = torch.tensor(list(m.values()), dtype=torch.float32)  # [M]

        coords_tensor = torch.from_numpy(coords.T).unsqueeze(0)  # [1,3,P]
        metrics_tensor = metrics_vec.unsqueeze(0)  # [1,M]
        return coords_tensor, metrics_tensor, fname


def collate_fn_test(batch):
    """
    Batch of tuples: (coords [1,3,P_i], metrics [1,M_i], filename)
    Pads to coords [B,1,3,Pmax], metrics [B,1,Mmax].
    """
    coords_list, metrics_list, filenames = zip(*batch)
    B = len(batch)

    max_points = max((c.shape[-1] for c in coords_list), default=0)
    max_m = max((m.shape[-1] for m in metrics_list if m.numel() > 0), default=0)

    padded_coords = torch.zeros((B, 1, 3, max_points), dtype=torch.float32)
    padded_metrics = torch.zeros((B, 1, max_m), dtype=torch.float32)

    for i in range(B):
        c = coords_list[i]  # [1,3,P_i] or [1,3,0]
        m = metrics_list[i]  # [1,M_i] or [1,0]
        if c.shape[-1] > 0:
            P_i = c.shape[-1]
            padded_coords[i, 0, :, :P_i] = c[0]
        if m.shape[-1] > 0:
            M_i = m.shape[-1]
            padded_metrics[i, 0, :M_i] = m[0]

    return {"coords": padded_coords, "metrics": padded_metrics, "filenames": filenames}
