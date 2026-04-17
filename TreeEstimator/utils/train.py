import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.SpeciesEstimation import EnsembleTreeEstimator, TreeEstimator
from models.SpeciesEstimationMetrics import TreeEstimatorMetrics
from models.SpeciesEstimationPoint import TreeEstimatorPoint
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from utils.loss_utils import ce_loss, center_loss
from utils.send_telegram import send_photos, send_telegram
from utils.tools import (
    AdaptiveLRScheduler,
    collate_fn,
    make_confusion_matrix,
    plot_gradients,
    plot_loss,
)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_params, "Trainable": trainable_params}


def train(params, train_set, test_set):
    if params["n_gpus"] > 1:
        device = torch.device("cuda" if params["cuda"] else "cpu")
    else:
        device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")

    exp_name = params["exp_name"]

    # Define Model
    if params["n_ensemble"] > 1:
        model = EnsembleTreeEstimator(
            params["num_species"],
            params["n_metrics"],
            params["layers"],
            params["n_ensemble"],
            params["first_dim"],
            params["last_dim"],
            extractor=params["extractor"],
        ).to(device)
    if params["model"] == "TreeEstimator":
        model = TreeEstimator(
            params["num_species"],
            params["first_dim"],
            params["last_dim"],
            params["layers"],
            params["n_metrics"],
            params["extractor"],
        ).to(device)
    if params["model"] == "TreeEstimatorMetrics":
        model = TreeEstimatorMetrics(
            num_species=params["num_species"], input_dim=params["n_metrics"]
        ).to(device)
    if params["model"] == "TreeEstimatorPoint":
        model = TreeEstimatorPoint(
            params["num_species"],
            params["first_dim"],
            params["last_dim"],
            params["layers"],
            params["extractor"],
        ).to(device)

    param_counts = count_parameters(model)
    print(param_counts)


#     # Run in Parallel
#     if params["n_gpus"] > 1:
#         model = nn.DataParallel(
#             model.cuda(), device_ids=list(range(0, params["n_gpus"]))
#         )

#     # Set up optimizers
#     if params["optimizer"] == "sgd":
#         optimizer = optim.SGD(
#             model.parameters(),
#             lr=params["lr"],
#             momentum=params["momentum"],
#             weight_decay=1e-4,
#         )
#     elif params["optimizer"] == "adam":
#         optimizer = optim.Adam(
#             model.parameters(), lr=params["lr"], betas=(0.9, 0.999), eps=1e-08
#         )
#     else:
#         raise Exception("Optimizer Not Implemented")

#     if params["adaptive_lr"] is True:
#         scheduler = AdaptiveLRScheduler(
#             optimizer,
#             patience=3,  # reduce LR if no improvement for 3 epochs
#             factor=0.5,  # reduce LR by a factor of 0.5
#             min_lr=1e-5,  # switch to cosine when reaching this lr
#             switch_threshold=1.2,  # threshold to switch slightly above min_lr
#             T_0=10,  # first cosine cycle lasts 10 epochs
#             T_mult=2,  # each restart cycle is twice as long
#         )

#     best_test_loss = np.inf
#     triggertimes = 0

#     train_weights = (
#         params["train_weights"].to(device)
#         if params["train_weights"] is not None
#         else None
#     )
#     val_weights = (
#         params["val_weights"].to(device) if params["val_weights"] is not None else None
#     )

#     early_stopping_patience = params.get("early_stopping_patience", 30)
#     epochs_no_improve = 0

#     test_loader = DataLoader(
#         test_set,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         pin_memory=False,
#         collate_fn=lambda x: collate_fn(x),
#     )

#     # Adaptive hard sample mining parameters
#     if params["hard_mining"]:
#         start_hard_ratio = params.get("start_hard_ratio", 1.0)  # 100% in early epochs
#         final_hard_ratio = params.get("final_hard_ratio", 0.25)
#         hard_ratio_decay_epochs = params.get(
#             "hard_ratio_decay_epochs", 100
#         )  # Linear decay

#     for epoch in tqdm(
#         range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"
#     ):
#         torch.cuda.empty_cache()
#         train_loss = 0.0
#         count = 0
#         train_true = []
#         train_pred = []

#         if params["hard_mining"]:
#             current_hard_ratio = start_hard_ratio - (
#                 start_hard_ratio - final_hard_ratio
#             ) * (epoch / hard_ratio_decay_epochs)
#             current_hard_ratio = max(final_hard_ratio, current_hard_ratio)

#         # Optionally shuffle indices or adjust train_set as needed
#         rem = len(train_set) % (params["batch_size"] * params["n_gpus"])
#         if rem != 0:
#             train_set = Subset(train_set, range(len(train_set) - rem))

#         train_loader = DataLoader(
#             train_set,
#             batch_size=params["batch_size"],
#             shuffle=True,
#             pin_memory=True,
#             collate_fn=lambda x: collate_fn(x),
#         )

#         for data, metrics, label, filename in tqdm(
#             train_loader, desc="Training Total: ", leave=False, colour="cyan"
#         ):
#             data, metrics, label = (
#                 data.to(device),
#                 metrics.to(device),
#                 label.to(device).squeeze(),
#             )
#             batch_size = data.size(0)

#             model.train()
#             optimizer.zero_grad()

#             if params["model"] == "TreeEstimatorMetrics":
#                 species, probs, logits, metrics_feats = model(metrics)
#             elif params["model"] == "TreeEstimatorPoint":
#                 species, probs, logits, x_feats = model(data)
#             else:
#                 species, probs, logits, x_feats, metrics_feats = model(data, metrics)

#             if params["hard_mining"]:
#                 # Compute per-sample losses for optimization and logging
#                 loss_ce = ce_loss(logits, label, train_weights, hard_mining=True)
#                 loss_ce_full = ce_loss(logits, label, train_weights, False).mean()
#                 loss_x_feats = center_loss(
#                     label, x_feats, params["num_species"], x_feats.size(1), True
#                 )
#                 loss_x_feats_full = center_loss(
#                     label, x_feats, params["num_species"], x_feats.size(1), False
#                 ).mean()
#                 loss_metrics_feats = center_loss(
#                     label,
#                     metrics_feats,
#                     params["num_species"],
#                     metrics_feats.size(1),
#                     True,
#                 )
#                 loss_metrics_feats_full = center_loss(
#                     label,
#                     metrics_feats,
#                     params["num_species"],
#                     metrics_feats.size(1),
#                     False,
#                 ).mean()

#                 losses = loss_ce + (loss_x_feats * 0.5) + (loss_metrics_feats * 0.5)
#                 loss_full = (
#                     loss_ce_full
#                     + (loss_x_feats_full * 0.5)
#                     + (loss_metrics_feats_full * 0.5)
#                 )

#                 # Select adaptive hard samples (remove .detach() to allow gradients)
#                 k_hard = max(1, int(current_hard_ratio * batch_size))
#                 hard_losses, hard_indices = torch.topk(losses, k_hard)

#                 # Randomly select 25% from the remaining samples
#                 all_indices = torch.arange(batch_size, device=device)
#                 mask = torch.ones(batch_size, dtype=torch.bool, device=device)
#                 mask[hard_indices] = False
#                 remaining_indices = all_indices[mask]
#                 if remaining_indices.numel() > 0:
#                     k_rand = max(1, int(0.25 * remaining_indices.numel()))
#                     rand_perm = torch.randperm(
#                         remaining_indices.numel(), device=device
#                     )[:k_rand]
#                     rand_indices = remaining_indices[rand_perm]
#                     random_losses = losses[rand_indices]
#                     combined_losses = torch.cat([hard_losses, random_losses])
#                 else:
#                     combined_losses = hard_losses

#                 loss = combined_losses.mean()
#             else:
#                 if params["model"] == "TreeEstimatorMetrics":
#                     loss_ce = ce_loss(logits, label, train_weights).mean()
#                     loss_metrics_feats = center_loss(
#                         label,
#                         metrics_feats,
#                         params["num_species"],
#                         metrics_feats.size(1),
#                     ).mean()
#                     loss = loss_ce + loss_metrics_feats
#                     loss_full = loss
#                 elif params["model"] == "TreeEstimatorPoint":
#                     loss_ce = ce_loss(logits, label, train_weights).mean()
#                     loss_x_feats = center_loss(
#                         label, x_feats, params["num_species"], x_feats.size(1)
#                     ).mean()
#                     loss = loss_ce + loss_x_feats
#                     loss_full = loss
#                 else:
#                     loss_ce = ce_loss(logits, label, train_weights).mean()
#                     loss_x_feats = center_loss(
#                         label, x_feats, params["num_species"], x_feats.size(1)
#                     ).mean()
#                     loss_metrics_feats = center_loss(
#                         label,
#                         metrics_feats,
#                         params["num_species"],
#                         metrics_feats.size(1),
#                     ).mean()
#                     loss = loss_ce + (loss_x_feats * 0.5) + (loss_metrics_feats * 0.5)
#                     loss_full = loss

#             loss.backward()
#             torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
#             optimizer.step()

#             # Log the full batch loss for consistency across training strategies
#             train_loss += loss_full.item() * batch_size
#             count += batch_size

#             train_true.append(label.cpu().numpy())
#             train_pred.append(species.cpu().numpy())

#         train_true = np.concatenate(train_true)
#         train_pred = np.concatenate(train_pred)
#         train_f1 = f1_score(train_true, train_pred, average="weighted")

#         train_loss = float(train_loss) / count

#         model.eval()
#         with torch.no_grad():
#             masks = False
#             test_loss = 0.0
#             count = 0
#             test_pred = []
#             test_pred_probs = []  # Will store a list of probability vectors
#             test_true = []
#             all_masks = []

#             for data, metrics, label, filename in tqdm(
#                 test_loader, desc="Validation Total: ", leave=False, colour="green"
#             ):
#                 data, metrics, label = (
#                     data.to(device),
#                     metrics.to(device),
#                     label.to(device).squeeze(),
#                 )
#                 batch_size = data.size(0)

#                 if params["model"] == "TreeEstimatorMetrics":
#                     species, probs, logits, masks, metrics_feats = model(metrics, True)
#                 elif params["model"] == "TreeEstimatorPoint":
#                     species, probs, logits, x_feats = model(data)
#                 else:
#                     (species, probs, logits, masks, x_feats, metrics_feats) = model(
#                         data, metrics, True
#                     )
#                 if masks is not False:
#                     all_masks.append(masks.cpu())

#                 if params["model"] == "TreeEstimatorMetrics":
#                     loss_ce = ce_loss(logits, label, val_weights).mean()
#                     loss_metrics_feats = center_loss(
#                         label,
#                         metrics_feats,
#                         params["num_species"],
#                         metrics_feats.size(1),
#                     ).mean()
#                     loss = loss_ce + loss_metrics_feats
#                 elif params["model"] == "TreeEstimatorPoint":
#                     loss_ce = ce_loss(logits, label, val_weights).mean()
#                     loss_x_feats = center_loss(
#                         label, x_feats, params["num_species"], x_feats.size(1)
#                     ).mean()
#                     loss = loss_ce + loss_x_feats
#                 else:
#                     loss_ce = ce_loss(logits, label, val_weights).mean()
#                     loss_x_feats = center_loss(
#                         label, x_feats, params["num_species"], x_feats.size(1)
#                     ).mean()
#                     loss_metrics_feats = center_loss(
#                         label,
#                         metrics_feats,
#                         params["num_species"],
#                         metrics_feats.size(1),
#                     ).mean()
#                     loss = loss_ce + (loss_x_feats * 0.5) + (loss_metrics_feats * 0.5)

#                 count += batch_size
#                 test_loss += loss.item() * batch_size

#                 test_true.append(label.cpu().numpy())
#                 test_pred.append(species.cpu().numpy())
#                 # Save the full probability vector for each sample in the batch.
#                 test_pred_probs.extend(probs.cpu().numpy().tolist())

#             test_true = np.concatenate(test_true)
#             test_pred = np.concatenate(test_pred)
#             test_f1 = f1_score(test_true, test_pred, average="weighted")
#             test_loss = float(test_loss) / count

#             if masks is not False:
#                 all_masks = torch.cat(all_masks, dim=0)
#                 feature_importance = all_masks.mean(dim=(0, 1))
#                 feature_importance = feature_importance.numpy()

#         out_dict = {
#             "epoch": [epoch + 1],
#             "train_loss": [train_loss],
#             "train_f1": [train_f1],
#             "val_loss": [test_loss],
#             "val_f1": [test_f1],
#         }
#         out_df = pd.DataFrame.from_dict(out_dict)

#         if epoch + 1 > 1:
#             loss_f1_df = pd.read_csv(f"checkpoints/{exp_name}/loss_f1.csv")
#             loss_f1_df = pd.concat([loss_f1_df, out_df])
#             loss_f1_df.to_csv(f"checkpoints/{exp_name}/loss_f1.csv", index=False)
#             loss_f1_df = pd.read_csv(f"checkpoints/{exp_name}/loss_f1.csv")
#             plot_loss(
#                 loss_f1_df,
#                 title=f"{exp_name}: Training and Validation Loss",
#                 save_path=f"checkpoints/{exp_name}/losses.png",
#             )
#         else:
#             out_df.to_csv(f"checkpoints/{exp_name}/loss_f1.csv", index=False)
#             plot_loss(
#                 out_df,
#                 title=f"{exp_name}: Training and Validation Loss",
#                 save_path=f"checkpoints/{exp_name}/losses.png",
#             )

#         # Apply addaptive learning
#         if params["adaptive_lr"] is True:
#             scheduler.step(loss)
#             # scheduler.step()
#             print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']}")

#         if test_loss < best_test_loss:
#             best_test_loss = test_loss
#             best_epoch = epoch + 1
#             epochs_no_improve = 0
#             torch.save(
#                 model.state_dict(), f"checkpoints/{exp_name}/models/best_model.t7"
#             )

#             # Create a DataFrame for predictions
#             out_df = pd.DataFrame(
#                 {
#                     "y_true": test_true,
#                     "y_pred": test_pred,
#                 }
#             )

#             # Assume the number of classes equals the length of the first probability vector.
#             if test_pred_probs:
#                 num_classes = len(test_pred_probs[0])
#                 prob_df = pd.DataFrame(
#                     test_pred_probs, columns=[f"prob_{i}" for i in range(num_classes)]
#                 )
#                 out_df = pd.concat([out_df, prob_df], axis=1)

#             out_df.to_csv(f"checkpoints/{exp_name}/output/output.csv", index=False)
#             if masks is not False:
#                 feature_df = pd.DataFrame(
#                     {
#                         "Feature Index": np.arange(len(feature_importance)),
#                         "Importance Score": feature_importance,
#                     }
#                 )

#                 feature_df.to_csv(
#                     f"checkpoints/{exp_name}/output/feature_importance.csv", index=False
#                 )

#         else:
#             epochs_no_improve += 1
#             print(f"Epochs with no improvement: {epochs_no_improve}")
#             if epochs_no_improve >= early_stopping_patience:
#                 print("Early Stopping")
#                 try:
#                     send_telegram("Early Stopping")
#                 except:
#                     pass
#                 break

#         print(
#             f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Training f1: {train_f1}, Validation Loss: {test_loss}, Validation f1: {test_f1}, Best loss: {best_test_loss} at Epoch {best_epoch}"
#         )

#         try:
#             send_telegram(f"{exp_name}: Epoch: {epoch+1}, Validation f1: {test_f1}")
#             send_photos(open(f"checkpoints/{exp_name}/losses.png", "rb"))
#         except:
#             pass


def test(params, test_set):
    device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Define Model
    if params["model"] == "TreeEstimator":
        model = TreeEstimator(
            params["num_species"],
            params["first_dim"],
            params["last_dim"],
            params["layers"],
            params["n_metrics"],
            params["extractor"],
        ).to(device)
    if params["model"] == "TreeEstimatorMetrics":
        model = TreeEstimatorMetrics(
            num_species=params["num_species"], input_dim=params["n_metrics"]
        ).to(device)

    if params["model"] == "TreeEstimatorPoint":
        model = TreeEstimatorPoint(
            params["num_species"],
            params["first_dim"],
            params["last_dim"],
            params["layers"],
            params["extractor"],
        ).to(device)

    # Load weights/biases
    state_dict = torch.load(params["model_path"], map_location=device)
    model.load_state_dict(state_dict)

    test_loader = DataLoader(
        test_set,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=False,
        collate_fn=lambda x: collate_fn(x),
    )

    model.eval()
    os.makedirs(f"checkpoints/{exp_name}/output/testing", exist_ok=True)

    with torch.no_grad():
        test_pred = []
        test_pred_probs = []  # Will store a list of probability vectors
        test_true = []
        all_masks = []

        masks = False
        for data, metrics, label, filename in tqdm(
            test_loader, desc="Validation Total: ", leave=False, colour="green"
        ):
            data, metrics, label = (
                data.to(device),
                metrics.to(device),
                label.to(device).squeeze(),
            )
            batch_size = data.size(0)

            if params["model"] == "TreeEstimatorMetrics":
                species, probs, logits, masks, metrics_feats = model(metrics, True)
            elif params["model"] == "TreeEstimatorPoint":
                species, probs, logits, x_feats = model(data)
            else:
                (species, probs, logits, masks, x_feats, metrics_feats) = model(
                    data, metrics, True
                )
            if masks is not False:
                all_masks.append(masks.cpu())

            test_true.append(label.cpu().numpy())
            test_pred.append(species.cpu().numpy())
            # Save the full probability vector for each sample in the batch.
            test_pred_probs.extend(probs.cpu().numpy().tolist())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_f1 = f1_score(test_true, test_pred, average="weighted")
        if masks is not False:
            all_masks = torch.cat(all_masks, dim=0)
            feature_importance = all_masks.mean(dim=(0, 1))
            feature_importance = feature_importance.numpy()

        # Create a DataFrame for predictions
        out_df = pd.DataFrame(
            {
                "y_true": test_true,
                "y_pred": test_pred,
            }
        )

        # Assume the number of classes equals the length of the first probability vector.
        if test_pred_probs:
            num_classes = len(test_pred_probs[0])
            prob_df = pd.DataFrame(
                test_pred_probs, columns=[f"prob_{i}" for i in range(num_classes)]
            )
            out_df = pd.concat([out_df, prob_df], axis=1)

        out_df.to_csv(f"checkpoints/{exp_name}/output/testing/output.csv", index=False)

        if masks is not False:
            feature_df = pd.DataFrame(
                {
                    "Feature Index": np.arange(len(feature_importance)),
                    "Importance Score": feature_importance,
                }
            )

            feature_df.to_csv(
                f"checkpoints/{exp_name}/output/testing/feature_importance.csv",
                index=False,
            )

        print(f"{exp_name}: F1: {test_f1}")
