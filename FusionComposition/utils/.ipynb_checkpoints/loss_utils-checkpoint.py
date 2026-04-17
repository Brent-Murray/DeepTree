import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_tensor(tensor, max_iterations=3):
    # Create a new tensor for adjustments
    adjusted_tensor = tensor.clone()

    # Round the tensor to the nearest tenth
    adjusted_tensor = adjusted_tensor.mul(10).round().div(10)

    for i in tqdm(range(adjusted_tensor.shape[0])):  # Iterate over the batch size
        for j in range(adjusted_tensor.shape[2]):  # Iterate over height
            for k in range(adjusted_tensor.shape[3]):  # Iterate over width
                iteration = 0
                while iteration < max_iterations:
                    iteration += 1
                    channel_tensor = adjusted_tensor[i, :, j, k].clone()
                    channel_sum = channel_tensor.sum()

                    if torch.isclose(channel_sum, torch.tensor(1.0), atol=1e-6):
                        break  # Sum is close to 1, no need to adjust

                    non_zero_indices = channel_tensor.nonzero(as_tuple=True)[0]
                    if channel_sum > 1.0:
                        # Subtract 0.1 from the max value
                        max_index = channel_tensor[non_zero_indices].argmax()
                        channel_tensor[non_zero_indices[max_index]] -= 0.1
                    else:
                        # Add 0.1 to the min value
                        min_index = channel_tensor[non_zero_indices].argmin()
                        channel_tensor[non_zero_indices[min_index]] += 0.1

                    # Update the adjusted tensor
                    adjusted_tensor[i, :, j, k] = channel_tensor

    return adjusted_tensor


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        # Create mask of no data values
        mask = y_true >= 0

        # Apply mask
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]

        if self.weights is None:
            squared_errors = torch.square(y_pred_masked - y_true_masked)
            loss = torch.sum(squared_errors) / torch.sum(mask)
        else:
            # Reshape weights to [C, 1, 1] to match the [B, C, H, W] shape of squared_errors
            self.weights = self.weights.view(-1, 1, 1)
            squared_errors = torch.square(y_pred_masked - y_true_masked)
            weighted_squared_errors = squared_errors * self.weights
            loss = torch.sum(weighted_squared_errors) / torch.sum(mask)

        return loss


class TopKAccuracyLoss(nn.Module):
    def __init__(self, k=3):
        super(TopKAccuracyLoss, self).__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        # Create Mask for negative values in y_true
        mask = y_true >= 0

        # Masked tensors
        y_pred_masked = torch.where(mask, y_pred, torch.tensor(float("nan")))
        y_true_masked = torch.where(mask, y_true, torch.tensor(float("nan")))

        # Get top k indices for each pixel
        _, pred_topk_idx = torch.topk(y_pred_masked, k=self.k, dim=1, largest=True)
        _, true_topk_idx = torch.topk(y_true_masked, k=self.k, dim=1, largest=True)

        # Calculate the number of correct values
        correct = torch.sum(
            pred_topk_idx == true_topk_idx, dim=[2, 3]
        )  # sum over H and W dimensions
        correct = correct.sum()  # sum over all B and C dimensions

        # calculate the number of total elements accounting for masked out values
        total_elements = torch.sum(mask) * self.k

        # Calculate loss
        loss = 1 - correct.float() / total_elements.float()

        return loss


def calc_loss(y_pred, y_true, weights=None, round_output=False, k=2):
    
    # Round Output
    if round_output:
        y_pred = adjust_tensor(y_pred)
    
    # Calculate weighted MSE
    weighted_mse = WeightedMSELoss(weights)
    mse_loss = weighted_mse(y_pred, y_true)

    # Calculate top k accuracy
    topk = TopKAccuracyLoss(k=k)
    topk_loss = topk(y_pred, y_true)

    # Calculate final loss
    loss = mse_loss + topk_loss

    return loss