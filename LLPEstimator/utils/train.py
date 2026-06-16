import os
import random
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.SpeciesLLP import SpeciesLLP
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from utils.loss_utils import LLPLoss
from utils.send_telegram import send_photos, send_telegram
from utils.tools import (
    AdaptiveLRScheduler,
    collate_fn,
    collate_fn_test,
    make_confusion_matrix,
    plot_gradients,
    plot_loss,
)


@torch.no_grad()
def bag_metrics_from_logits(logits, tree_mask, y_prop, eps=1e-8):
    """
    logits:   [B, T, C]
    tree_mask:[B, T]  (True for real trees)
    y_prop:   [B, C]  (plot-level target proportions; rows sum to 1)
    """
    B, T, C = logits.shape
    probs = F.softmax(logits, dim=-1)  # [B,T,C]

    # average only over real trees per plot
    mask = tree_mask.unsqueeze(-1).float()  # [B,T,1]
    summed = (probs * mask).sum(dim=1)  # [B,C]
    counts = mask.sum(dim=1).clamp_min(1.0)  # [B,1]
    p_hat = (summed / counts).clamp(min=eps, max=1.0)  # [B,C]
    p_tgt = y_prop.clamp(min=eps, max=1.0)  # [B,C]

    # MAE / MSE
    mae = (p_hat - p_tgt).abs().mean().item()
    mse = ((p_hat - p_tgt) ** 2).mean().item()

    # KL(target || pred) and JS
    kl = (p_tgt * (p_tgt.log() - p_hat.log())).sum(dim=1).mean().item()
    m = 0.5 * (p_tgt + p_hat)
    js = (
        (
            0.5 * (p_tgt * (p_tgt.log() - m.log())).sum(dim=1)
            + 0.5 * (p_hat * (p_hat.log() - m.log())).sum(dim=1)
        )
        .mean()
        .item()
    )

    # Pearson r per-class across plots, then mean
    def _pearson(a, b, dim=0, eps=1e-8):
        a = a - a.mean(dim, keepdim=True)
        b = b - b.mean(dim, keepdim=True)
        num = (a * b).sum(dim)
        den = (a.square().sum(dim) * b.square().sum(dim)).sqrt().clamp_min(eps)
        return num / den

    r_mean = torch.nanmean(_pearson(p_hat, p_tgt, dim=0)).item()

    # R² across all plots and species (flattened)
    ss_res = ((p_tgt - p_hat) ** 2).sum().item()
    ss_tot = ((p_tgt - p_tgt.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "prop_mae": mae,
        "prop_mse": mse,
        "kl_tgt_pred": kl,
        "js_div": js,
        "pearson_r_mean": r_mean,
        "r2_all": r2_all,
    }


def bag_metrics_from_props(plot_props, tree_mask, y_prop, eps=1e-8):
    """
    plot_props:   [B, C]
    tree_mask:[B, T]  (True for real trees)
    y_prop:   [B, C]  (plot-level target proportions; rows sum to 1)
    """
    p_hat = plot_props.clamp(min=eps, max=1.0)  # [B,C]
    p_tgt = y_prop.clamp(min=eps, max=1.0)  # [B,C]

    # MAE / MSE
    mae = (p_hat - p_tgt).abs().mean().item()
    mse = ((p_hat - p_tgt) ** 2).mean().item()

    # KL(target || pred) and JS
    kl = (p_tgt * (p_tgt.log() - p_hat.log())).sum(dim=1).mean().item()
    m = 0.5 * (p_tgt + p_hat)
    js = (
        (
            0.5 * (p_tgt * (p_tgt.log() - m.log())).sum(dim=1)
            + 0.5 * (p_hat * (p_hat.log() - m.log())).sum(dim=1)
        )
        .mean()
        .item()
    )

    # Pearson r per-class across plots, then mean
    def _pearson(a, b, dim=0, eps=1e-8):
        a = a - a.mean(dim, keepdim=True)
        b = b - b.mean(dim, keepdim=True)
        num = (a * b).sum(dim)
        den = (a.square().sum(dim) * b.square().sum(dim)).sqrt().clamp_min(eps)
        return num / den

    r_mean = torch.nanmean(_pearson(p_hat, p_tgt, dim=0)).item()

    # R² across all plots and species (flattened)
    ss_res = ((p_tgt - p_hat) ** 2).sum().item()
    ss_tot = ((p_tgt - p_tgt.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2_all = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "prop_mae": mae,
        "prop_mse": mse,
        "kl_tgt_pred": kl,
        "js_div": js,
        "pearson_r_mean": r_mean,
        "r2_all": r2_all,
    }


@torch.no_grad()
def plot_props_from_avg_tree_logits(logits, tree_mask, eps=1e-8):
    """
    logits:    [B, T, C]
    tree_mask: [B, T]
    returns:   plot_props [B, C] computed as softmax(mean_tree_logits)
    """
    B, T, C = logits.shape
    m = tree_mask.unsqueeze(-1).float()  # [B,T,1]
    n = m.sum(dim=1).clamp_min(1.0)  # [B,1]
    plot_logits = (logits * m).sum(dim=1) / n  # [B,C]
    plot_props = F.softmax(plot_logits, dim=-1)  # [B,C]
    return plot_props.clamp(min=eps, max=1.0)


def train(params, train_set, val_set):
    # Define Modeling Parameters
    device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]
    best_val_loss = np.inf  # set initially to infinity
    early_stopping_patience = params.get("early_stopping_patience", 30)
    epochs_no_improve = 0

    # Model Set Up
    model = SpeciesLLP(
        params["num_species"],
        params["first_dim"],
        params["last_dim"],
        params["layers"],
        params["n_metrics"],
        params["mn_hidden"],
        params["mn_out"],
        params["feature_dim"],
        use_aggregator=params["use_aggregator"],
        plot_prop_mode=params["plot_prop_mode"],
    ).to(device)

    # Set Up Optimizer and LR Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=params["lr"], betas=(0.9, 0.999), eps=1e-08
    )

    scheduler = AdaptiveLRScheduler(
        optimizer,
        patience=3,  # reduce LR if no improvement for 3 epochs
        factor=0.5,  # reduce LR by a factor of 0.5
        min_lr=1e-5,  # switch to cosine when reaching this lr
        switch_threshold=1.2,  # threshold to switch slightly above min_lr
        T_0=10,  # first cosine cycle lasts 10 epochs
        T_mult=2,  # each restart cycle is twice as long
    )

    # Set Up Train/Val Weights
    train_weights = (
        params["train_weights"].to(device)
        if params["train_weights"] is not None
        else None
    )

    val_weights = (
        params["val_weights"].to(device) if params["val_weights"] is not None else None
    )

    # Load Validaiton Data
    val_loader = DataLoader(
        val_set,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    train_loss_fn = LLPLoss(weights=train_weights, lambda_aux=0.1, beta=2.0).to(device)
    val_loss_fn = LLPLoss(weights=val_weights, lambda_aux=0.1, beta=2.0).to(device)

    # Train Through Epochs
    for epoch in tqdm(
        range(params.get("epochs", 300)),
        desc="Model Total: ",
        leave=False,
        colour="red",
    ):
        torch.cuda.empty_cache()
        train_loss = 0.0
        count = 0
        train_true = []
        train_pred = []

        # Shuffle indices and adjust train_set as needed
        if params["subset"]:
            subset_size = int(0.01 * len(train_set))
            train_subset, _ = random_split(
                train_set, [subset_size, len(train_set) - subset_size]
            )
            indices = np.arange(len(train_subset))
        else:
            indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        rem = len(indices) % params["batch_size"]
        if rem != 0:
            indices = indices[:-rem]
            if params["subset"]:
                train_subset = Subset(train_subset, indices)
            else:
                train_set = Subset(train_set, indices)

        # Load Training Data
        if params["subset"]:
            train_loader = DataLoader(
                train_subset,
                batch_size=params["batch_size"],
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=params["batch_size"],
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        # Iterate Through Train Loader
        for batch in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            coords = batch["coords"].to(device)
            metrics = batch["metrics"].to(device)
            targets = batch["targets"].to(device).squeeze()

            batch_size = coords.size(0)

            # Train Model
            model.train()
            optimizer.zero_grad()
            # logits = model(coords, metrics, treeid_maps)
            logits, tree_mask, plot_props = model(
                coords, metrics, return_plot_props=True
            )
            # loss = calc_loss(
            #     logits, plot_props, targets, tree_mask, train_weights
            # ).mean()
            loss = train_loss_fn(logits, plot_props, targets, tree_mask)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update loss and count
            train_loss += loss.item() * batch_size
            count += batch_size

            # *** CHANGE WHAT OUTPUT WILL BE ***
            # Append train true and pred lists
            # train_true.append(targets.cpu().numpy())
            # train_pred.append(logits.detach().cpu().numpy())

        # Concatenate train true and pred lists
        # train_true = np.concatenate(train_true)
        # train_pred = np.concatenate(train_pred)

        # Get Epoch Train Loss
        train_loss = float(train_loss) / count

        # Set up validation loop
        model.eval()
        with torch.no_grad():
            masks = False
            val_loss = 0.0
            count = 0
            val_pred = []
            val_true = []
            all_masks = []
            all_filenames = []
            metric_sums_tree = defaultdict(float)
            metric_sums_plot = defaultdict(float)
            pred_rows = []

            # Iterate through validation data
            for batch in tqdm(
                val_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                coords = batch["coords"].to(device)
                metrics = batch["metrics"].to(device)
                targets = batch["targets"].to(device).squeeze()
                filenames = batch["filenames"]
                all_filenames.append(filenames)

                batch_size = coords.size(0)

                # Validate Model
                logits, tree_mask, plot_props, pred_maps = model(
                    coords, metrics, return_plot_props=True, return_preds=True
                )
                # loss = calc_loss(
                #     logits, plot_props, targets, tree_mask, val_weights
                # ).mean()
                loss = val_loss_fn(logits, plot_props, targets, tree_mask)

                for b in range(batch_size):
                    fname = filenames[b]

                    for t, cls_id in pred_maps[b].items():
                        pred_rows.append(
                            {
                                "filename": fname,
                                "TreeID": int(t),
                                "Class_id": int(cls_id),
                            }
                        )

                bag_metrics_tree = bag_metrics_from_logits(logits, tree_mask, targets)
                for k, v in bag_metrics_tree.items():
                    metric_sums_tree[k] += float(v) * batch_size

                bag_metrics_plot = bag_metrics_from_props(
                    plot_props, tree_mask, targets
                )
                for k, v in bag_metrics_plot.items():
                    metric_sums_plot[k] += float(v) * batch_size

                # Append masks
                if masks is not False:
                    all_masks.append(masks)

                # Update loss and count
                val_loss += loss.item() * batch_size
                count += batch_size

                # *** CHANGE WHAT OUTPUT WILL BE ***
                # Append val true and pred lists
                val_true.append(targets.cpu().numpy())
                val_pred.append(plot_props.detach().cpu().numpy())

            # concatenate true and pred lists
            val_true = np.concatenate(val_true)
            val_pred = np.concatenate(val_pred)

            # Get epoch val loss
            val_loss = float(val_loss) / count

            if masks is not False:
                all_masks = torch.cat(all_masks, dim=0)
                feature_importance = all_masks.mean(dim=(0, 1, 2))
                feature_importance = feature_importance.cpu().numpy()
                feature_importance = np.ravel(feature_importance)

            pred_df = pd.DataFrame(pred_rows)
            pred_df.to_csv(
                f"checkpoints/{exp_name}/output/species_predictions/preds_epoch_{epoch+1:03d}.csv",
                index=False,
            )

        val_bag_metrics_tree = {
            k: (s / max(1, count)) for k, s in metric_sums_tree.items()
        }
        val_bag_metrics_plot = {
            k: (s / max(1, count)) for k, s in metric_sums_plot.items()
        }

        # Apply Adaptive Learning
        scheduler.step(loss)
        print(f"\nCurrent LR: {scheduler.optimizer.param_groups[0]['lr']}")

        # Create dataframe of training/validation metrics
        out_dict = {
            "epoch": [epoch + 1],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            **{f"tree_{k}": [v] for k, v in val_bag_metrics_tree.items()},
            **{f"plot_{k}": [v] for k, v in val_bag_metrics_plot.items()},
        }
        out_df = pd.DataFrame.from_dict(out_dict)

        if epoch + 1 > 1:
            loss_df = pd.read_csv(f"checkpoints/{exp_name}/loss.csv")
            loss_df = pd.concat([loss_df, out_df])
            loss_df.to_csv(f"checkpoints/{exp_name}/loss.csv", index=False)
        else:
            out_df.to_csv(f"checkpoints/{exp_name}/loss.csv", index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                model.state_dict(), f"checkpoints/{exp_name}/models/best_model.t7"
            )

            out_df = pd.DataFrame(
                {
                    "y_true": np.ravel(val_true),
                    "y_pred": np.ravel(val_pred),
                }
            )
            out_df.to_csv(f"checkpoints/{exp_name}/output/output.csv", index=False)

            if masks is not False:
                feature_df = pd.DataFrame(
                    {
                        "Feature Index": np.arange(len(feature_importance)),
                        "Importance Score": feature_importance,
                    }
                )

                feature_df.to_csv(
                    f"checkpoints/{exp_name}/output/feature_importance.csv", index=False
                )

        else:
            epochs_no_improve += 1
            print(f"Epochs with no improvement: {epochs_no_improve}")
            if epochs_no_improve >= early_stopping_patience:
                print("Early Stopping")
                try:
                    send_telegram("Early Stopping")
                except:
                    pass
                break
        print(
            f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Best Loss: {best_val_loss} at Epoch {best_epoch}"
        )
        print(
            "val_tree | "
            + " | ".join([f"{k}: {v:.4f}" for k, v in val_bag_metrics_tree.items()])
        )
        print(
            "val_plot | "
            + " | ".join([f"{k}: {v:.4f}" for k, v in val_bag_metrics_plot.items()])
        )

        try:
            send_telegram(
                f"{exp_name}: Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
            )
            # send_photos(open(f"checkpoints/{exp_name}/losses.png", "rb"))
        except:
            pass


# def test(
#     params,
#     test_set,
#     save_feature_csv=True,
#     save_reductions=True,
#     do_tsne=True,
#     do_umap=True,
#     max_points=50000,
#     random_state=42,
# ):
#     """
#     Runs eval on test_set, returns (feature_importance_np, all_masks_tensor)
#       - feature_importance_np: shape [M], averaged over all trees and all steps
#       - all_masks_tensor:      shape [N_valid_trees, n_steps, M] on CPU
#     """
#     device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")
#     exp_name = params["exp_name"]

#     # Build model
#     model = SpeciesLLP(
#         params["num_species"],
#         params["first_dim"],
#         params["last_dim"],
#         params["layers"],
#         params["n_metrics"],
#         params["mn_hidden"],
#         params["mn_out"],
#         params["feature_dim"],
#         use_aggregator=params["use_aggregator"],
#         plot_prop_mode=params["plot_prop_mode"],
#     ).to(device)
#     model.eval()

#     checkpoint_path = params["model_path"]
#     state = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # Data loader
#     test_loader = DataLoader(
#         test_set,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         pin_memory=False,
#         collate_fn=collate_fn,
#     )

#     all_masks_cpu = []
#     feat_list, pred_list = [], []

#     # Helper: per-tree embedding
#     def per_tree_embed_and_logits(coords_b, metrics_b):
#         point_mask = coords_b.abs().sum(dim=1) > 0
#         tree_mask = point_mask.any(dim=1)
#         valid_t = torch.nonzero(tree_mask, as_tuple=True)[0].tolist()
#         feats, logits = [], []
#         for t in valid_t:
#             pm = point_mask[t]
#             pts = coords_b[t, :, pm]
#             met = metrics_b[t]
#             pts_feat = model.point_extractor(pts)
#             met_feat = model.metrics_extractor(met)
#             if pts_feat.dim() > 1:
#                 pts_feat = pts_feat.squeeze(0)
#             if met_feat.dim() > 1:
#                 met_feat = met_feat.squeeze(0)
#             fused = torch.cat([pts_feat, met_feat], dim=-1)
#             fused = model.fusion(fused)
#             logits.append(model.species_head(fused).unsqueeze(0))
#             feats.append(fused.unsqueeze(0))
#         if feats:
#             return torch.cat(feats), torch.cat(logits)
#         return torch.empty(0, params["feature_dim"]), torch.empty(
#             0, params["num_species"]
#         )

#     with torch.no_grad():
#         for batch in tqdm(
#             test_loader, desc="Test (inference)", leave=False, colour="green"
#         ):
#             coords = batch["coords"].to(device)
#             metrics = batch["metrics"].to(device)
#             logits, tree_mask, pred_maps = model(coords, metrics, return_preds=True)
#             if tree_mask.any():
#                 valid_metrics = metrics[tree_mask]
#                 _, masks = model.metrics_extractor(valid_metrics, return_masks=True)
#                 all_masks_cpu.append(masks.detach().cpu())
#             B = coords.size(0)
#             for b in range(B):
#                 feats_b, logits_b = per_tree_embed_and_logits(coords[b], metrics[b])
#                 if feats_b.numel() == 0:
#                     continue
#                 preds_b = torch.argmax(logits_b, dim=-1)
#                 feat_list.append(feats_b.cpu())
#                 pred_list.append(preds_b.cpu())

#     # Aggregate masks → feature importance
#     if len(all_masks_cpu) == 0:
#         feature_importance = np.zeros(params["n_metrics"], dtype=np.float32)
#         all_masks = torch.empty(
#             (0, getattr(model.metrics_extractor, "n_steps", 0), params["n_metrics"])
#         )
#     else:
#         all_masks = torch.cat(all_masks_cpu, dim=0)
#         feature_importance = all_masks.mean(dim=(0, 1)).numpy()

#     if save_feature_csv:
#         pd.DataFrame(
#             {
#                 "Feature Index": np.arange(len(feature_importance)),
#                 "Importance Score": feature_importance,
#             }
#         ).to_csv(
#             os.path.join("checkpoints", exp_name, "output/feature_importance_test.csv"),
#             index=False,
#         )

#     # Combine embeddings
#     if feat_list:
#         feats = torch.cat(feat_list, dim=0).numpy()
#         preds = torch.cat(pred_list, dim=0).numpy()
#     else:
#         feats = np.empty((0, params["feature_dim"]))
#         preds = np.empty((0,), dtype=int)

#     embeddings = {"features": feats, "pred_ids": preds}

#     # Optionally save t-SNE / UMAP
#     if save_reductions and feats.shape[0] > 1:
#         rng = np.random.RandomState(random_state)
#         if feats.shape[0] > max_points:
#             idx = rng.choice(feats.shape[0], size=max_points, replace=False)
#             X, y = feats[idx], preds[idx]
#         else:
#             X, y = feats, preds

#         results = {"pred_ids": y}

#         if do_tsne:
#             from sklearn.manifold import TSNE

#             tsne = TSNE(
#                 n_components=3,
#                 perplexity=min(30, max(5, X.shape[0] // 50)),
#                 learning_rate="auto",
#                 init="pca",
#                 random_state=random_state,
#             )
#             results["tsne"] = tsne.fit_transform(X)

#         if do_umap:
#             try:
#                 import umap

#                 um = umap.UMAP(
#                     n_components=2,
#                     n_neighbors=15,
#                     min_dist=0.1,
#                     metric="euclidean",
#                     random_state=random_state,
#                 )
#                 results["umap"] = um.fit_transform(X)
#             except ImportError:
#                 print("UMAP not installed; skipping.")

#         np.savez(
#             os.path.join("checkpoints", exp_name, "output/latent_reductions.npz"),
#             **results,
#         )


# def test(
#     params,
#     test_set,
#     save_feature_csv=True,
#     save_tree_logits_csv=True,
#     tree_logits_filename="tree_logits_test.csv",
# ):
#     """
#     Runs eval on test_set, returns (feature_importance_np, all_masks_tensor, embeddings, tree_logits_df)

#       - feature_importance_np: shape [M], averaged over all trees and all steps
#       - all_masks_tensor:      shape [N_valid_trees, n_steps, M] on CPU
#       - embeddings:            {"features": np.ndarray [N, feature_dim], "pred_ids": np.ndarray [N]}
#       - tree_logits_df:        pandas DataFrame with filepath, tree_id, pred_id, and logits_*
#     """
#     device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")
#     exp_name = params["exp_name"]

#     # Build model
#     model = SpeciesLLP(
#         params["num_species"],
#         params["first_dim"],
#         params["last_dim"],
#         params["layers"],
#         params["n_metrics"],
#         params["mn_hidden"],
#         params["mn_out"],
#         params["feature_dim"],
#         use_aggregator=params["use_aggregator"],
#         plot_prop_mode=params["plot_prop_mode"],
#     ).to(device)
#     model.eval()

#     checkpoint_path = params["model_path"]
#     state = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(state)
#     model.eval()

#     # Data loader
#     test_loader = DataLoader(
#         test_set,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         pin_memory=False,
#         collate_fn=collate_fn,
#     )

#     all_masks_cpu = []
#     feat_list, pred_list = [], []
#     tree_rows = []  # <- per-tree logits + metadata rows

#     # Helper: per-tree embedding (kept)
#     def per_tree_embed_and_logits(coords_b, metrics_b):
#         point_mask = coords_b.abs().sum(dim=1) > 0
#         tree_mask = point_mask.any(dim=1)
#         valid_t = torch.nonzero(tree_mask, as_tuple=True)[0].tolist()
#         feats, logits = [], []
#         for t in valid_t:
#             pm = point_mask[t]
#             pts = coords_b[t, :, pm]
#             met = metrics_b[t]
#             pts_feat = model.point_extractor(pts)
#             met_feat = model.metrics_extractor(met)
#             if pts_feat.dim() > 1:
#                 pts_feat = pts_feat.squeeze(0)
#             if met_feat.dim() > 1:
#                 met_feat = met_feat.squeeze(0)
#             fused = torch.cat([pts_feat, met_feat], dim=-1)
#             fused = model.fusion(fused)
#             logits.append(model.species_head(fused).unsqueeze(0))
#             feats.append(fused.unsqueeze(0))
#         if feats:
#             return torch.cat(feats), torch.cat(logits)
#         return torch.empty(0, params["feature_dim"]), torch.empty(
#             0, params["num_species"]
#         )

#     with torch.no_grad():
#         for batch in tqdm(
#             test_loader, desc="Test (inference)", leave=False, colour="green"
#         ):
#             coords = batch["coords"].to(device)
#             metrics = batch["metrics"].to(device)

#             # logits: [B, T, K] ; tree_mask: [B, T]
#             logits, tree_mask, pred_maps = model(coords, metrics, return_preds=True)

#             # ---- feature masks -> importance
#             if tree_mask.any():
#                 valid_metrics = metrics[tree_mask]
#                 _, masks = model.metrics_extractor(valid_metrics, return_masks=True)
#                 all_masks_cpu.append(masks.detach().cpu())

#             # ---- try to pull filepath + optional explicit tree ids from batch
#             # filepath candidates: "filepath", "filepaths", "path", "paths"
#             filepaths = (
#                 batch.get("filepaths", None)
#                 or batch.get("filepath", None)
#                 or batch.get("paths", None)
#                 or batch.get("path", None)
#             )
#             # normalize to a Python list length B if possible
#             if isinstance(filepaths, str):
#                 filepaths = [filepaths]
#             elif torch.is_tensor(filepaths):
#                 # if someone stored as tensor of strings, keep as-is (rare)
#                 filepaths = [str(x) for x in filepaths]

#             # tree id candidates: "tree_ids" (preferred) or "tree_id"
#             # expected shape [B, T] or list-of-lists
#             tree_ids = batch.get("tree_ids", None) or batch.get("tree_id", None)

#             B, T, K = logits.shape
#             for b in range(B):
#                 fp = None
#                 if isinstance(filepaths, (list, tuple)) and b < len(filepaths):
#                     fp = filepaths[b]

#                 # valid tree indices for this plot
#                 valid_t = torch.nonzero(tree_mask[b], as_tuple=True)[0].tolist()
#                 for t in valid_t:
#                     # choose tree id: explicit if provided, else use index t
#                     if tree_ids is None:
#                         tid = t
#                     else:
#                         if torch.is_tensor(tree_ids):
#                             tid = int(tree_ids[b, t].item())
#                         elif isinstance(tree_ids, (list, tuple)):
#                             # could be list-of-lists or list-of-tensors
#                             tid = tree_ids[b][t]
#                             if torch.is_tensor(tid):
#                                 tid = int(tid.item())
#                             else:
#                                 tid = int(tid)
#                         else:
#                             tid = t

#                     logit_vec = logits[b, t].detach().cpu().numpy()  # [K]
#                     pred_id = int(np.argmax(logit_vec))

#                     row = {
#                         "filepath": fp,
#                         "tree_id": tid,
#                         "pred_id": pred_id,
#                     }
#                     # store logits as separate columns for easy CSV usage
#                     for k in range(K):
#                         row[f"logit_{k}"] = float(logit_vec[k])
#                     tree_rows.append(row)

#             # ---- keep embeddings output as before (per-tree fused feature + argmax on raw head)
#             for b in range(B):
#                 feats_b, logits_b = per_tree_embed_and_logits(coords[b], metrics[b])
#                 if feats_b.numel() == 0:
#                     continue
#                 preds_b = torch.argmax(logits_b, dim=-1)
#                 feat_list.append(feats_b.cpu())
#                 pred_list.append(preds_b.cpu())

#     # Aggregate masks → feature importance
#     if len(all_masks_cpu) == 0:
#         feature_importance = np.zeros(params["n_metrics"], dtype=np.float32)
#         all_masks = torch.empty(
#             (0, getattr(model.metrics_extractor, "n_steps", 0), params["n_metrics"])
#         )
#     else:
#         all_masks = torch.cat(all_masks_cpu, dim=0)
#         feature_importance = all_masks.mean(dim=(0, 1)).numpy()

#     if save_feature_csv:
#         pd.DataFrame(
#             {
#                 "Feature Index": np.arange(len(feature_importance)),
#                 "Importance Score": feature_importance,
#             }
#         ).to_csv(
#             os.path.join("checkpoints", exp_name, "output/feature_importance_test.csv"),
#             index=False,
#         )

#     # Combine embeddings
#     if feat_list:
#         feats = torch.cat(feat_list, dim=0).numpy()
#         preds = torch.cat(pred_list, dim=0).numpy()
#     else:
#         feats = np.empty((0, params["feature_dim"]))
#         preds = np.empty((0,), dtype=int)

#     embeddings = {"features": feats, "pred_ids": preds}

#     # Save / return per-tree logits table
#     tree_logits_df = pd.DataFrame(tree_rows)
#     if save_tree_logits_csv:
#         out_path = os.path.join(
#             "checkpoints", exp_name, f"output/{tree_logits_filename}"
#         )
#         tree_logits_df.to_csv(out_path, index=False)

#     return feature_importance, all_masks, embeddings, tree_logits_df


def test(
    params,
    test_set,
    save_feature_csv=True,
    save_tree_logits_csv=True,
    tree_logits_filename="tree_logits_test.csv",
    save_plot_props_csv=True,  # NEW
    plot_props_filename="plot_props_logits_avg_test.csv",  # NEW
):
    device = torch.device(f"cuda:{params['gpu_id']}" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    model = SpeciesLLP(
        params["num_species"],
        params["first_dim"],
        params["last_dim"],
        params["layers"],
        params["n_metrics"],
        params["mn_hidden"],
        params["mn_out"],
        params["feature_dim"],
        use_aggregator=params["use_aggregator"],
        plot_prop_mode=params["plot_prop_mode"],
    ).to(device)
    model.eval()

    checkpoint_path = params["model_path"]
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = DataLoader(
        test_set,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    all_masks_cpu = []
    feat_list, pred_list = [], []
    tree_rows = []
    plot_rows = []  # NEW: one row per plot with true/pred proportions (avg-logits)

    def per_tree_embed_and_logits(coords_b, metrics_b):
        point_mask = coords_b.abs().sum(dim=1) > 0
        tree_mask = point_mask.any(dim=1)
        valid_t = torch.nonzero(tree_mask, as_tuple=True)[0].tolist()
        feats, logits = [], []
        for t in valid_t:
            pm = point_mask[t]
            pts = coords_b[t, :, pm]
            met = metrics_b[t]
            pts_feat = model.point_extractor(pts)
            met_feat = model.metrics_extractor(met)
            if pts_feat.dim() > 1:
                pts_feat = pts_feat.squeeze(0)
            if met_feat.dim() > 1:
                met_feat = met_feat.squeeze(0)
            fused = torch.cat([pts_feat, met_feat], dim=-1)
            fused = model.fusion(fused)
            logits.append(model.species_head(fused).unsqueeze(0))
            feats.append(fused.unsqueeze(0))
        if feats:
            return torch.cat(feats), torch.cat(logits)
        return torch.empty(0, params["feature_dim"]), torch.empty(
            0, params["num_species"]
        )

    with torch.no_grad():
        for batch in tqdm(
            test_loader, desc="Test (inference)", leave=False, colour="green"
        ):
            coords = batch["coords"].to(device)
            metrics = batch["metrics"].to(device)

            # targets may or may not exist in your test loader
            targets = batch.get("targets", None)  # NEW
            if targets is not None:
                targets = targets.to(device).squeeze()

            # logits: [B,T,K] ; tree_mask: [B,T]
            logits, tree_mask, pred_maps = model(coords, metrics, return_preds=True)

            # NEW: plot proportions from avg tree logits (logits-mode aggregation)
            plot_props_logits_avg = plot_props_from_avg_tree_logits(
                logits, tree_mask
            )  # [B,K]

            # feature masks -> importance
            if tree_mask.any():
                valid_metrics = metrics[tree_mask]
                _, masks = model.metrics_extractor(valid_metrics, return_masks=True)
                all_masks_cpu.append(masks.detach().cpu())

            # filepath candidates
            filepaths = (
                batch.get("filepaths", None)
                or batch.get("filepath", None)
                or batch.get("paths", None)
                or batch.get("path", None)
            )
            if isinstance(filepaths, str):
                filepaths = [filepaths]
            elif torch.is_tensor(filepaths):
                filepaths = [str(x) for x in filepaths]

            tree_ids = batch.get("tree_ids", None) or batch.get("tree_id", None)

            B, T, K = logits.shape

            # NEW: write one row per plot (wide: true_k / pred_k)
            for b in range(B):
                fp = (
                    filepaths[b]
                    if isinstance(filepaths, (list, tuple)) and b < len(filepaths)
                    else None
                )

                pred_vec = plot_props_logits_avg[b].detach().cpu().numpy()  # [K]
                if targets is not None:
                    true_vec = targets[b].detach().cpu().numpy()  # [K]
                else:
                    true_vec = np.full((K,), np.nan, dtype=np.float32)

                row = {"filepath": fp}
                for k in range(K):
                    row[f"y_true_{k}"] = float(true_vec[k])
                for k in range(K):
                    row[f"y_pred_{k}"] = float(pred_vec[k])
                plot_rows.append(row)

            # per-tree logits CSV as before
            for b in range(B):
                fp = (
                    filepaths[b]
                    if isinstance(filepaths, (list, tuple)) and b < len(filepaths)
                    else None
                )
                valid_t = torch.nonzero(tree_mask[b], as_tuple=True)[0].tolist()
                for t in valid_t:
                    if tree_ids is None:
                        tid = t
                    else:
                        if torch.is_tensor(tree_ids):
                            tid = int(tree_ids[b, t].item())
                        elif isinstance(tree_ids, (list, tuple)):
                            tid = tree_ids[b][t]
                            tid = int(tid.item()) if torch.is_tensor(tid) else int(tid)
                        else:
                            tid = t

                    logit_vec = logits[b, t].detach().cpu().numpy()
                    pred_id = int(np.argmax(logit_vec))

                    row = {"filepath": fp, "tree_id": tid, "pred_id": pred_id}
                    for k in range(K):
                        row[f"logit_{k}"] = float(logit_vec[k])
                    tree_rows.append(row)

            # embeddings output as before
            for b in range(B):
                feats_b, logits_b = per_tree_embed_and_logits(coords[b], metrics[b])
                if feats_b.numel() == 0:
                    continue
                preds_b = torch.argmax(logits_b, dim=-1)
                feat_list.append(feats_b.cpu())
                pred_list.append(preds_b.cpu())

    # masks -> feature importance
    if len(all_masks_cpu) == 0:
        feature_importance = np.zeros(params["n_metrics"], dtype=np.float32)
        all_masks = torch.empty(
            (0, getattr(model.metrics_extractor, "n_steps", 0), params["n_metrics"])
        )
    else:
        all_masks = torch.cat(all_masks_cpu, dim=0)
        feature_importance = all_masks.mean(dim=(0, 1)).numpy()

    if save_feature_csv:
        pd.DataFrame(
            {
                "Feature Index": np.arange(len(feature_importance)),
                "Importance Score": feature_importance,
            }
        ).to_csv(
            os.path.join("checkpoints", exp_name, "output/feature_importance_test.csv"),
            index=False,
        )

    # embeddings
    if feat_list:
        feats = torch.cat(feat_list, dim=0).numpy()
        preds = torch.cat(pred_list, dim=0).numpy()
    else:
        feats = np.empty((0, params["feature_dim"]))
        preds = np.empty((0,), dtype=int)
    embeddings = {"features": feats, "pred_ids": preds}

    # per-tree logits CSV
    tree_logits_df = pd.DataFrame(tree_rows)
    if save_tree_logits_csv:
        out_path = os.path.join(
            "checkpoints", exp_name, f"output/{tree_logits_filename}"
        )
        tree_logits_df.to_csv(out_path, index=False)

    # NEW: per-plot props CSV (avg logits -> softmax)
    plot_props_df = pd.DataFrame(plot_rows)
    if save_plot_props_csv:
        out_path = os.path.join(
            "checkpoints", exp_name, f"output/{plot_props_filename}"
        )
        plot_props_df.to_csv(out_path, index=False)

    return feature_importance, all_masks, embeddings, tree_logits_df, plot_props_df


def load_model_from_ckpt(params, ckpt_path, device):
    model = SpeciesLLP(
        params["num_species"],
        params["first_dim"],
        params["last_dim"],
        params["layers"],
        params["n_metrics"],
        params["mn_hidden"],
        params["mn_out"],
        params["feature_dim"],
        use_aggregator=params["use_aggregator"],
        plot_prop_mode=params["plot_prop_mode"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# @torch.no_grad()
# def test_individual_trees(params, test_set, out_csv=None):
#     """
#     Runs per-tree inference on a dataset like PointCloudsInDF using your collate_fn.
#     Produces filename, TreeID (index within plot after filtering), and Class_id.
#     """
#     device = torch.device(
#         f"cuda:{params['gpu_id']}" if params.get("cuda", False) else "cpu"
#     )
#     exp_name = params["exp_name"]
#     checkpoint_path = params["model_path"]
#     if out_csv is None:
#         os.makedirs(f"checkpoints/{exp_name}/output/species_predictions", exist_ok=True)
#         out_csv = f"checkpoints/{exp_name}/output/species_predictions/test_preds.csv"

#     loader = DataLoader(
#         test_set,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         pin_memory=False,
#         collate_fn=collate_fn_test,  # <-- your existing collate_fn
#     )

#     model = load_model_from_ckpt(params, checkpoint_path, device)

#     rows = []
#     for batch in tqdm(
#         loader, desc="Test (individual trees)", leave=False, colour="green"
#     ):
#         coords = batch["coords"].to(device)  # [B, T, 3, P]
#         metrics = batch["metrics"].to(device)  # [B, T, M]
#         filenames = batch["filenames"]

#         # Forward with per-tree predictions
#         logits, tree_mask, pred_maps = model(coords, metrics, return_preds=True)

#         # Collect per-tree class ids (pred_maps[b] is {tree_index_in_batch: cls_id})
#         for b, fname in enumerate(filenames):
#             pm = pred_maps[b]  # dict[int,int]
#             if len(pm) == 0:
#                 # no valid trees in this plot after filtering/padding
#                 rows.append({"filename": fname, "TreeID": -1, "Class_id": -1})
#                 continue
#             for t, cls_id in pm.items():
#                 rows.append(
#                     {"filename": fname, "TreeID": int(t), "Class_id": int(cls_id)}
#                 )

#     pd.DataFrame(rows).to_csv(out_csv, index=False)
#     print(f"Wrote predictions to: {out_csv}")


@torch.no_grad()
def test_individual_trees(params, test_set, out_csv=None):
    """
    Runs per-tree inference and writes:
      filename, TreeID, Class_id, prob_0..prob_{K-1}
    """
    device = torch.device(
        f"cuda:{params['gpu_id']}" if params.get("cuda", False) else "cpu"
    )
    exp_name = params["exp_name"]
    checkpoint_path = params["model_path"]
    K = params["num_species"]

    if out_csv is None:
        os.makedirs(f"checkpoints/{exp_name}/output/species_predictions", exist_ok=True)
        out_csv = f"checkpoints/{exp_name}/output/species_predictions/test_preds.csv"

    loader = DataLoader(
        test_set,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn_test,
    )

    model = load_model_from_ckpt(params, checkpoint_path, device)

    rows = []

    def _append_nan_rows_for_filenames(fnames):
        for fname in fnames:
            row = {"filename": fname, "TreeID": -1, "Class_id": -1}
            for k in range(K):
                row[f"prob_{k}"] = float("nan")
            rows.append(row)

    for batch in tqdm(
        loader, desc="Test (individual trees)", leave=False, colour="green"
    ):
        torch.cuda.empty_cache()
        filenames = batch["filenames"]

        try:
            coords = batch["coords"].to(device)  # [B, T, 3, P]
            metrics = batch["metrics"].to(device)  # [B, T, M]

            # Forward
            logits, tree_mask, pred_maps = model(
                coords, metrics, return_preds=True
            )  # logits [B,T,K]
            probs = F.softmax(logits, dim=-1)  # [B,T,K]

        except RuntimeError as e:
            # Handle CUDA OOM (and continue with NaNs)
            if "out of memory" in str(e).lower():
                print("⚠️ CUDA OOM — skipping batch and writing NaNs")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _append_nan_rows_for_filenames(filenames)
                continue
            raise  # any other RuntimeError should surface

        # Normal processing
        for b, fname in enumerate(filenames):
            pm = pred_maps[b]  # dict[int,int] mapping tree-index -> class-id
            if len(pm) == 0:
                row = {"filename": fname, "TreeID": -1, "Class_id": -1}
                for k in range(K):
                    row[f"prob_{k}"] = float("nan")
                rows.append(row)
                continue

            for t, cls_id in pm.items():
                p = probs[b, t].detach().cpu().numpy()  # [K]
                row = {"filename": fname, "TreeID": int(t), "Class_id": int(cls_id)}
                for k in range(K):
                    row[f"prob_{k}"] = float(p[k])
                rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote predictions to: {out_csv}")
