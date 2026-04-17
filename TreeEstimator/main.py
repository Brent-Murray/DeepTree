import logging
import os
import sys
import warnings
from ast import literal_eval
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from utils.augmentation import AugmentPointCloudsInDF
from utils.pointcloud_metrics import calculate_metrics
from utils.send_telegram import send_telegram
from utils.tools import PointCloudsInDF, _init_
from utils.train import test, train

warnings.filterwarnings("ignore")


def main(params):
    _init_(params["exp_name"])

    # Load Datasets
    df_path = params["df_path"]
    df = pd.read_csv(df_path)

    train_df = df[df["split"] == "train"]  # training dataframe
    val_df = df[df["split"] == "validation"]  # validation dataframe
    test_df = df[df["split"] == "test"]  # validation dataframe

    data_path = params["data_path"]
    trainset = PointCloudsInDF(data_path, train_df)

    if params["augment"] == True:
        classes = train_df["class"].unique().tolist()
        train_count = [len(train_df[train_df["class"] == i]) for i in classes]
        n_augs = [round((max(train_count) - i) / i) for i in train_count]
        aug_trainset = AugmentPointCloudsInDF(data_path, train_df)
        trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])
    else:
        pass

    if params["train_weights"] == True:
        # Count the frequency of each class in the training set
        class_counts = Counter(train_df["class"])
        # Ensure weights are ordered by class label (if needed)
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[0])
        counts = [count for _, count in sorted_counts]
        total = sum(counts)
        # Compute weights as the inverse proportion of the class count (scaled by 100)
        class_weights = [1 / (100 * (n / total)) for n in counts]
        params["train_weights"] = torch.tensor(class_weights)
    else:
        params["train_weights"] = None

    if params["val_weights"] == True:
        # Count the frequency of each class in the validation set
        class_counts = Counter(val_df["class"])
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[0])
        counts = [count for _, count in sorted_counts]
        total = sum(counts)
        class_weights = [1 / (100 * (n / total)) for n in counts]
        params["val_weights"] = torch.tensor(class_weights)
    else:
        params["val_weights"] = None

    if not params["eval"]:
        valset = PointCloudsInDF(data_path, val_df)
        train(params, trainset, valset)
    else:
        testset = PointCloudsInDF(data_path, test_df)
        test(params, testset)


if __name__ == "__main__":
    params = {  # EdgeConv + TabNet
        "exp_name": "SpeciesEstimator_13",  # experiment name
        "batch_size": 12,  # batch size 6 for ensemble 12 for others
        "df_path": r"D:\MurrayBrent\projects\paper3\data\raw\RMF_ITD\trees.csv",  # df path
        "data_path": r"D:\MurrayBrent\projects\paper3\data\raw\RMF_ITD\las",
        "augment": False,  # augment
        "train_weights": True,  # training weights
        "val_weights": True,  # validation weights
        "hard_mining": False,  # hard training mining
        "n_gpus": 1,  # number of gpus
        "epochs": 300,  # total epochs
        "optimizer": "adam",  # optimizer
        "lr": 1e-3,  # learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "early_stopping_patience": 20,  # early stopping patience
        "momentum": 0.9,  # sgd momentum
        "model_path": r"D:\MurrayBrent\projects\paper3\scripts\SpeciesEstimator\checkpoints\SpeciesEstimator_8\models\best_model.t7",  # pretrained model path
        "num_species": 4,  # number of species
        "n_metrics": 16,  # number of metrics used for TabNet
        "first_dim": 64,  # first layer output dimension for PointExtractor
        "last_dim": 128,  # last layer output dimension for PointExtractor
        "layers": 4,  # number of layers in PointExtractor
        "extractor": "pointtransformer",  # feature extractor model
        "cuda": True,  # use cuda
        "gpu_id": 0,  # gpu id to use
        "eval": False,  # run testing
        "model": "TreeEstimator",  # model to run
        "n_ensemble": 1,
    }
    mn = params["exp_name"]
    print(f"Starting {mn}")
    send_telegram(f"starting {mn}")
    main(params)
