import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from tqdm import tqdm

class IOStream:
    # Adapted from https://github.com/vinits5/learning3d/blob/master/examples/train_pointnet.py
    def __init__(self, path):
        # Open file in append
        self.f = open(path, "a")

    def cprint(self, text):
        # Print and write text to file
        print(text)  # print text
        self.f.write(text + "\n")  # write text and new line
        self.f.flush  # flush file

    def close(self):
        self.f.close()  # close file

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
    if not os.path.exists("checkpoints/" + model_name + "/output/figures"):
        os.makedirs("checkpoints/" + model_name + "/output/figures")
    if not os.path.exists("checkpoints/" + model_name + "/output/test_rasters"):
        os.makedirs("checkpoints/" + model_name + "/output/test_rasters")


def tensor_to_geotiff(
    raster_tensor,
    file_name,
    reference_raster_path,
    label_tensor,
    dtype=rasterio.float32,
):
    """
    Convert a PyTorch tensor to a GeoTIFF file using the metadata from a reference raster.

    :param tensor: PyTorch tensor to convert.
    :param file_name: Output GeoTIFF file name.
    :param reference_raster_path: Path to the reference raster file to copy metadata from.
    :param dtype: Data type for the rasterio writer, default is rasterio.float32.
    """
    # Create mask of no data values
    mask = label_tensor >= 0
    
    raster_tensor_masked = raster_tensor.masked_fill(~mask, -9999)

    # Convert the PyTorch tensor to a numpy array
    array = raster_tensor_masked.cpu().numpy()

    # Ensure the tensor is in the correct shape (bands, rows, columns)
    if len(array.shape) == 2:
        array = array[np.newaxis, :, :]

    # Read the metadata from the reference raster
    with rasterio.open(reference_raster_path) as ref_rst:
        ref_meta = ref_rst.meta.copy()

    # Update the metadata with the new count and dtype from the tensor
    ref_meta.update({"count": array.shape[0], "dtype": dtype, "driver": "GTiff"})

    # Write the array data to a GeoTIFF file
    with rasterio.open(file_name, "w", **ref_meta) as dst:
        for i in range(array.shape[0]):
            dst.write(array[i], i + 1)
            
            
class R2Score(nn.Module):
    def __init__(self):
        super(R2Score, self).__init__()

    def forward(self, y_pred, y_true):
        # Flatten tensors while preserving batch dimension
        # New shape will be [batch_size, -1]
        y_true_flat = y_true.view(y_true.shape[0], -1)
        y_pred_flat = y_pred.view(y_pred.shape[0], -1)

        # Create mask for flattened tensors
        mask = y_true_flat >= 0

        # Apply mask
        y_true_masked = torch.masked_select(y_true_flat, mask)
        y_pred_masked = torch.masked_select(y_pred_flat, mask)

        # Calculate R2 score
        sst = torch.sum((y_true_masked - torch.mean(y_true_masked)) ** 2)
        ssr = torch.sum((y_true_masked - y_pred_masked) ** 2)
        r2_score = 1 - ssr / sst

        return r2_score
    
def loss_figure(csv_path, out_path):
    # Set options
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["axes.linewidth"] = 1

    # Read loss csv
    df = pd.read_csv(csv_path)

    # Find lowest loss
    best_loss_epoch = df["epoch"].iloc[df["val_loss"].idxmin()] - 1
    best_val_loss = df["val_loss"].iloc[df["val_loss"].idxmin()]

    # Plot loss values
    plt.plot(df["train_loss"], label="Training loss", color="blue")  # train loss
    plt.plot(df["val_loss"], label="Validation loss", color="orange")  # val loss

    # Plot best lines
    plt.axvline(
        x=best_loss_epoch, label="Lowest Validation loss", color="r", ls=":"
    )  # epoch
    plt.axhline(y=best_val_loss, color="r", ls=":")  # loss
    
    # Position the text with an offset and new line
    offset = 0.01
    text_x = best_loss_epoch + (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * offset
    text_y = best_val_loss + (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * offset
    text = f"Epoch: {best_loss_epoch}\nLoss: {best_val_loss:.4f}"
    plt.text(text_x, text_y, text, color="r", ha="left", va="bottom")
    
    # Add axis labels and legend
    plt.xlabel("Epoch")  # x axis
    plt.ylabel("Loss")  # y axis
    plt.legend()  # legend
    
    plt.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close()
    
    
def adjust_tensor(tensor, max_iterations=10):
    # Round the tensor to the nearest tenth
    tensor = torch.round(tensor * 10) / 10

    for i in range(tensor.shape[0]):  # Iterate over the batch size
        for j in range(tensor.shape[2]):  # Iterate over height
            for k in range(tensor.shape[3]):  # Iterate over width
                iteration = 0
                while iteration < max_iterations:
                    iteration += 1
                    channel_sum = tensor[i, :, j, k].sum()

                    if torch.isclose(channel_sum, torch.tensor(1.0), atol=1e-6):
                        break  # Sum is close to 1, no need to adjust

                    # Find the index to adjust
                    non_zero_indices = tensor[i, :, j, k].nonzero().flatten()
                    if channel_sum > 1.0:
                        # Subtract 0.1 from the max value
                        max_index = tensor[i, :, j, k][non_zero_indices].argmax()
                        tensor[i, non_zero_indices[max_index], j, k] -= 0.1
                    else:
                        # Add 0.1 to the min value
                        min_index = tensor[i, :, j, k][non_zero_indices].argmin()
                        tensor[i, non_zero_indices[min_index], j, k] += 0.1

                    # Correct for rounding errors
                    tensor[i, :, j, k] = torch.clamp(tensor[i, :, j, k], min=0, max=1)
                    tensor[i, :, j, k] = torch.round(tensor[i, :, j, k] * 10) / 10

    return tensor