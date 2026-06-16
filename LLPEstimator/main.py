import os
import sys
import warnings
from ast import literal_eval
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.send_telegram import send_telegram
from utils.tools import PointCloudsInDF, SingleTreeDataset, _init_
from utils.train import test, test_individual_trees, train

warnings.filterwarnings("ignore")


def main(params):
    _init_(params["exp_name"])

    # Load Datasets and Run Model
    df_path = params["df_path"]
    df = pd.read_csv(df_path)
    if params["test_trees"]:
        pass
    else:
        df["perc_specs"] = df["perc_specs"].apply(literal_eval)

    data_path = params["data_path"]

    # Testing
    if params["eval"]:
        val_df = df[df["split"] == "validation"]  # validation dataframe
        testset = PointCloudsInDF(data_path, val_df)  # load testset
        test(params, testset)  # run test
    elif params["test_trees"]:
        testset = SingleTreeDataset(data_path, df["filename"])
        test_individual_trees(params, testset)

    # Training
    else:
        train_df = df[df["split"] == "train"]  # training dataframe
        val_df = df[df["split"] == "validation"]  # validation dataframe
        trainset = PointCloudsInDF(data_path, train_df)  # load trainset
        valset = PointCloudsInDF(data_path, val_df)  # load valset

        # Update Training Weights
        if params["train_weights"] == True:
            non_zero_indices = [
                index
                for row in train_df["perc_specs"]
                for index, value in enumerate(row)
                if value != 0
            ]
            index_counts = Counter(non_zero_indices)
            index_counts = [int(index_counts[k]) for k in sorted(index_counts.keys())]
            class_weights = [1 / (100 * (n / sum(index_counts))) for n in index_counts]
            params["train_weights"] = torch.tensor(class_weights)
        else:
            params["train_weights"] = None

        # Update Validation Weights
        if params["val_weights"] == True:
            non_zero_indices = [
                index
                for row in val_df["perc_specs"]
                for index, value in enumerate(row)
                if value != 0
            ]
            index_counts = Counter(non_zero_indices)
            index_counts = [int(index_counts[k]) for k in sorted(index_counts.keys())]
            class_weights = [1 / (100 * (n / sum(index_counts))) for n in index_counts]
            params["val_weights"] = torch.tensor(class_weights)
        else:
            params["val_weights"] = None

        train(params, trainset, valset)  # run training


if __name__ == "__main__":
    params = {
        "exp_name": "SpeciesLLP_twostage_updated",  # experiment name
        "cuda": True,  # use gpu
        "gpu_id": 0,  # gpu id
        "eval": False,  # run test
        "test_trees": True,
        "model_path": r"D:\MurrayBrent\git\SpeciesLLP\checkpoints\SpeciesLLP_twostage_updated\models\best_model.t7",  # pretrained model path
        "subset": False,  # subset training data
        "batch_size": 6,  # batch size
        "df_path": r"D:\MurrayBrent\projects\paper4\data\rmf_tiles_itd\trees.csv",  # df path
        "data_path": r"D:\MurrayBrent\projects\paper4\data\rmf_tiles_itd\laz",  # dataset path
        # "df_path": r"D:\MurrayBrent\projects\paper4\data\rmf_itd\trees.csv",  # df path
        # "data_path": r"D:\MurrayBrent\projects\paper4\data\rmf_itd\las",  # dataset path
        # "df_path": r"D:\MurrayBrent\projects\paper4\data\rmf_plots\photo_plots.csv",  # df path
        # "data_path": r"D:\MurrayBrent\projects\paper4\data\rmf_plots\segmented_laz",  # dataset path
        "train_weights": True,  # use training weights
        "val_weights": True,  # use validaiton weights
        "epochs": 300,  # number of training epochs
        "lr": 1e-3,  # initial learning rate
        "adaptive_lr": True,  # use adaptive learning rate
        "early_stopping_patience": 10,  # early stopping patience
        "num_species": 9,  # number of species
        "n_metrics": 16,  # number of input metrics for MetricsNetwork
        "mn_hidden": 64,  # hidden dimension for MetricsNetwork
        "mn_out": 128,  # output dimensions for MetricsNetwork
        "first_dim": 16,  # first layer dimension for PointExtractor
        "last_dim": 64,  # last layer dimension for PointExtractor
        "layers": 4,  # number of PointExtractor layers
        "feature_dim": 256,  # output fused features dimension
        "use_aggregator": False,  # use attention based aggregator
        "plot_prop_mode": "features",
    }

    mn = params["exp_name"]
    print(f"Starting {mn}")
    send_telegram(f"Starting {mn}")
    main(params)
