import os
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.data import PointCloudsInPickle, RastersInDF
from utils.send_telegram import send_telegram
from utils.tools import IOStream, _init_
from utils.train import test, test_dgcnn, train

def transfer_dgcnn(params):
    _init_(params["exp_name"])

    # Initiate IOStream
    mn = params["exp_name"]
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(f"Starting {mn}")

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Read Dataset
    test_data_path = params["test_path"]  # test dataset path
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle, "id", "label")
    print("Validating Files")
    testset.validate_files()
    
    # Run Model
    test_dgcnn(params, io, testset)


def main(params):
    # Set up folder structure
    _init_(params["exp_name"])

    # Initiate IOStream
    mn = params["exp_name"]
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(f"Starting {mn}")

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Read datasets
    trainset = RastersInDF(
        params["raster_folder"], params["label_folder"], params["train_df"]
    )  # read training dataset

    # Augment Dataset
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = RastersInDF(
                params["raster_folder"],
                params["label_folder"],
                params["train_df"],
                augment=True,
            )
            trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])
            

    valset = RastersInDF(
        params["raster_folder"], params["label_folder"], params["val_df"]
    )  # read validation dataset

    testset = RastersInDF(
        params["raster_folder"], params["label_folder"], params["test_df"]
    )  # read testing dataset

    # Run training/testing
    if not params["test"]:
        train(params, io, trainset, valset)  # train model
        torch.cuda.empty_cache()  # empty cache
    else:
        test(params, testset)  # test model


def find_key(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None  # Return None if the target value is not found in the dictionary


def check_nonzero_bands(raster_file, labels_dict):
    try:
        # Get the filename from the provided path
        filename = os.path.basename(raster_file)

        # Open the raster file
        with rasterio.open(raster_file) as src:
            # Initialize a dictionary to store band results
            band_results = {"Filename": filename}

            # Loop through each band
            for band_idx in range(1, src.count + 1):
                band = src.read(band_idx)

                # Get the nodata value for the band
                nodata = src.nodatavals[0]

                # Count non-zero and non-nodata values in the band
                nonzero_count = ((band != 0) & (band != nodata)).sum()

                # Store the result in the dictionary
                band_label = find_key(labels_dict, band_idx)
                band_results[band_label] = nonzero_count

            # Create a DataFrame from the dictionary
            df = pd.DataFrame([band_results])

            return df

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Make sure to close the raster file even in case of an error
        src.close()


def calculate_weights(bands_labels, training_df, raster_folder):
    dfs = []
    for img in tqdm(
        training_df["Filename"].to_list(),
        desc="Checking Bands",
        colour="yellow",
        leave=False,
    ):
        img = os.path.join(raster_folder, img)
        df = check_nonzero_bands(img, bands_labels)
        if not df.empty:
            dfs.append(df)
    if dfs:
        concatenated_df = pd.concat(dfs, ignore_index=True)

    dfs = []
    for key, _ in tqdm(
        bands_labels.items(), desc="Calculating Weights", colour="yellow", leave=False
    ):
        sum_value = sum(concatenated_df[key])
        count_value = len(concatenated_df[concatenated_df[key] != 0])

        # Create a DataFrame for the current key
        df = pd.DataFrame({"Key": [key], "Sum": [sum_value], "Count": [count_value]})

        # Append it to the list of DataFrames
        dfs.append(df)

    # Concatenate all DataFrames into one
    result_df = pd.concat(dfs, ignore_index=True)
    count = result_df["Count"].to_list()
    sums = result_df["Sum"].to_list()

    return count, sums


if __name__ == "__main__":
    # Get Dataframes
    df = pd.read_csv(r"D:\MurrayBrent\projects\paper2\data\processed\tiles_128_split.csv")
    train_df = df[df["split"] == "train"]  # training dataframe
    val_df = df[df["split"] == "validation"]  # validation dataframe
    test_df = df[df["split"] == "test"]  # testing dataframe
    # test_df = df

    # Calculate Training weights
    bands_labels = {
        "BF": 1,
        "BW": 2,
        "CE": 3,
        "LA": 4,
        "PJ": 5,
        "PO": 6,
        "PT": 7,
        "SB": 8,
        "SW": 9,
    }
    count, sums = calculate_weights(
        bands_labels, train_df, r"D:\MurrayBrent\projects\paper2\data\processed\labels\tiles_128"
    )
    class_weights = [1 / (100 * (n / sum(count))) for n in count]  # per class weights
    weights = torch.tensor(class_weights)  # weights as tensor

    # Data folder
    data_folder = r"D:\MurrayBrent\projects\paper2\data\processed\RMF_S2"
    # seasons = ["fall", "summer", "spring", "winter"]
    seasons = ["fall", "summer"]
    raster_folders = [os.path.join(data_folder, season, "tiles_128") for season in seasons]
    # raster_folders.append(r"D:\MurrayBrent\projects\paper2\data\processed\DEM\tiles_128")
    raster_folders.append(r"D:\MurrayBrent\projects\paper2\data\processed\DGCNN\tiles_128")
    
    # Define parameters
    params = {
        "exp_name": "retain_unet_fusion_wcat_k2_bap",  # experiment name
        "model": "retain_unet",  # model
        # "model": "fusion_unet",
        "join": "attention", # join type 
        "raster_folder": raster_folders,  # raster folder
        "label_folder": r"D:\MurrayBrent\projects\paper2\data\processed\labels\tiles_128",  # labels folder
        "train_df": train_df,  # training dataframe
        "val_df": val_df,  # validation dataframe
        "test_df": test_df,  # testing dataframe
        "test_path": r"F:\paper2\resampled", # path to dgcnn data
        "test_pickle": r"D:\MurrayBrent\projects\paper2\data\raw\RMF_SPL\RMF_plots\pixel_center.pkl", # path to dgcnn pickle
        "model_path": r"D:\MurrayBrent\projects\paper2\scripts\checkpoints\retain_unet_fusion_wcat_k2\models\best_model.pt",  # path to trained model
        # "model_path": r"D:\MurrayBrent\projects\paper2\scripts\checkpoints\fusion_attention_unet_weights_k1\models\best_model.pt",
        "augment": True, # augment training data
        "n_augs": 5,  # number of augmentations
        "batch_size": 24,  # batch size
        "classes": ["BF", "BW", "CE", "LA", "PJ", "PO", "PT", "SB", "SW"],  # classes
        "num_models": len(seasons), # number of image based inputs
        "in_channels": 9,  # number of input channels (bands)
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": 200,  # total number of epochs
        "optimizer": "adam",  # optimizer
        "momentum": 0.9,  # sgd momentum
        "train_weights": weights,  # training weights
        "lr": 1e-3,  # learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": 20,  # patience for adaptive learning rate
        "step_size": 50,  # step size for adaptive learning rate
        "dropout": 0.5,  # dropout rate
        "k": 2, # number of top k for loss
        "cuda": True,  # use cuda
        "test": False,  # run testing
        "train_weights": weights,  # training weights
        "telegram_token": None,  # telegram token
        "telegram_id": None,  # telegram chat id
    }

    # Initiate model
    mn = params["exp_name"]  # model name
    token = params["telegram_token"]
    chat_id = params["telegram_id"]
    if params["test"]:
        send_telegram(f"Testing {mn}", token, chat_id)  # send telegram
    else:
        send_telegram(f"Training {mn}", token, chat_id)  # send telegram

    # Run model
    # transfer_dgcnn(params)
    main(params)