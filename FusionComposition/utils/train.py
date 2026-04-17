import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.dgcnn import DGCNN
from models.fusion_unet import FusionUNet
from models.unet import UNet
from models.ensamble_unet import EnsambleUNet
from models.retain_unet import RetFuseNet
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from utils.loss_utils import calc_loss
from utils.send_telegram import send_telegram, send_photos
from utils.tools import R2Score, loss_figure, tensor_to_geotiff
from sklearn.metrics import r2_score as sk_r2

warnings.filterwarnings("ignore")

def custom_collate_fn(batch):
    """
    Custom collate function to filter out None data points in the batch.
    """
    # First, check if the batch itself is None
    if batch is None:
        return None
    
    # Next, filter out any top-level None items in the batch
    filtered_batch = [item for item in batch if item is not None]
    
    # Now, filter out items where any of item[0], item[1], or item[2] are None
    filtered_batch = [item for item in filtered_batch if all(x is not None for x in item)]

    # If after filtering, the batch is empty, handle accordingly
    if len(filtered_batch) == 0:
        return None

    # Use the default collate function on the filtered batch
    return default_collate(filtered_batch)

def train(params, io, train_set, val_set):
    # Define Device & Experiment Name
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Set Up Model
    if params["model"] == "fusion_unet":  # Fusion Unet
        model = FusionUNet(in_channels=params["in_channels"], num_classes=len(params["classes"]),  num_models=params["num_models"], join=params['join']).to(device).cuda()
    elif params["model"] == "unet":  # Unet
        model = (
            UNet(in_channels=params["in_channels"], num_classes=len(params["classes"]))
            .to(device)
            .cuda()
        )
    elif params["model"] == "ensamble_unet": # ensamble unet
        model = (
            EnsambleUNet(in_channels=params["in_channels"], num_classes=len(params["classes"]), join=params['join'])
            .to(device)
            .cuda()
        )
    elif params["model"] == "retain_unet":
        model = RetFuseNet(in_channels=params["in_channels"], num_classes=len(params["classes"]), num_models=params["num_models"], join=params['join']).to(device).cuda()
    else:
        model_name = params["model"]
        raise Exception(f"Model: {model_name} Not Implemented")

    # Run in Parallel
    if params["n_gpus"] > 1:
        model = nn.DataParallel(
            model.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )

    # Set Up Optimizers
    if params["optimizer"] == "sgd":  # sgd optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer"] == "adam":  # adam optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=params["lr"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        optimizer_name = params["optimizer"]
        raise Exception(f"Optimizer: {optimizer_name} Not Implemented")

    # Set Up Adaptive Learning
    if params["adaptive_lr"] is True:
        # Scheduler 1: Reduce on plateau
        scheduler1 = ReduceLROnPlateau(optimizer, "min", patience=params["patience"])

        # Scheduler 2: Step
        scheduler2 = StepLR(optimizer, step_size=params["step_size"], gamma=0.1)

        change = 0  # initial change value

    # Set Initial Loss Value
    best_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0

    # # Set Training Weights
    # if params["train_weights"] is not None:
    #     train_weights = params["train_weights"].to(device)
    # else:
    #     train_weights = None

    val_loader = DataLoader(
        val_set, batch_size=params["batch_size"], shuffle=False, pin_memory=True
    )

    # Train model for set number of epochs
    for epoch in tqdm(
        range(params["epochs"]), desc="Model Total", leave=False, colour="red"
    ):
        # Set training weights
        if epoch+1 >= 25:
            if params["train_weights"] is not None:
                train_weights = params["train_weights"].to(device)
            else:
                train_weights = None
        else:
            train_weights = None
        
        # Randomize Training Set
        trainset_idx = list(range(len(train_set)))  # list idx
        random.shuffle(trainset_idx)  # shuffle idx
        rem = (
            len(trainset_idx) % params["batch_size"]
        )  # find number of idx to remove to fit in gpus
        if rem <= 3:
            trainset_idx = trainset_idx[: len(trainset_idx) - rem]  # remove idx
            train_set = Subset(train_set, trainset_idx)

        # Load Train Set
        train_loader = DataLoader(
            train_set, batch_size=params["batch_size"], shuffle=True, pin_memory=True
        )

        # Empty Containers
        train_losses = []
        train_r2s = []

        # Iterate through loader
        for data, label, filename in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            data, label = (
                # data.to(device),
                {k: v.to(device) for k, v in data.items()},
                label.to(device).squeeze(),
            )


            # Start Training
            model.train()  # Initiate Training
            output = model(data)  # Run Model
            optimizer.zero_grad()  # zero_gradients
            loss = calc_loss(F.softmax(output, dim=1), label, train_weights, k=params["k"])
            loss.backward()  # Backpropogation
            optimizer.step()  # optimize

            # Calculate RScore
            r2_score = R2Score()
            r2 = r2_score(F.softmax(output, dim=1), label)

            # Append scores
            train_losses.append(loss.item())
            train_r2s.append(r2.item())

        # Calculate mean r2 and loss
        train_r2 = np.mean(train_r2s)
        train_loss = np.mean(train_losses)

        # Set Up Validation
        model.eval()
        with torch.no_grad():
            val_r2s = []
            val_losses = []

            # Validation
            for data, label, filename in tqdm(
                val_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                data, label = (
                    # data.to(device),
                    {k: v.to(device) for k, v in data.items()},
                    label.to(device).squeeze(),
                )

                # Run Model
                output = model(data)

                # Calculate Loss
                loss = calc_loss(F.softmax(output, dim=1), label, weights=None, k=params["k"])
                r2_score = R2Score()
                r2 = r2_score(F.softmax(output, dim=1), label)

                val_losses.append(loss.item())
                val_r2s.append(r2.item())

            # Get mean r2 and loss
            val_r2 = np.mean(val_r2s)
            val_loss = np.mean(val_losses)

        # Create output dataframe
        out_dict = {
            "epoch": [epoch + 1],
            "train_loss": [train_loss],
            "train_r2": [train_r2],
            "val_loss": [val_loss],
            "val_r2": [val_r2],
        }  # Create dictionary of values
        out_df = pd.DataFrame.from_dict(out_dict)  # Create dataframe from dictionary

        # Save Dataframe
        if epoch + 1 > 1:
            df = pd.read_csv(f"checkpoints/{exp_name}/loss_r2.csv")
            df = pd.concat([df, out_df])
            df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)
        else:
            out_df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)

        # If current loss beats best loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_loss_r2 = val_r2
            torch.save(
                model.state_dict(), f"checkpoints/{exp_name}/models/best_model.pt"
            )

        # Log and print loss/r2 values
        io.cprint(
            f"\nEpoch: {epoch +1}, Training Loss: {train_loss}, Training R2: {train_r2}, Validation Loss: {val_loss}, Validation R2: {val_r2}, Best loss: {best_loss} with {best_loss_r2} r2 at Epoch {best_epoch}"
        )

        # Create Loss figure
        loss_figure(
            f"checkpoints/{exp_name}/loss_r2.csv",
            f"checkpoints/{exp_name}/output/figures/loss_figure.png",
        )
        # Send Telegrams
        token = params["telegram_token"]
        chat_id = params["telegram_id"]
        try:
            if (epoch + 1) % 1 == 0:
                send_telegram(
                    f"Best Loss: {best_loss} with an R2: {best_loss_r2} at Epoch {best_epoch}",
                    token,
                    chat_id,
                )
                send_photos(
                    open(f"checkpoints/{exp_name}/output/figures/loss_figure.png", "rb"),
                    token,
                    chat_id,
                )
        except:
            pass

        # Apply Addaptive Learning
        if params["adaptive_lr"] is True:
            if val_loss > best_loss:
                triggertimes += 1
                if triggertimes == params["patience"]:
                    change = 1
            else:
                triggertimes = 0
            if change == 0:
                scheduler1.step(val_loss)
                current_lr = scheduler1.optimizer.param_groups[0]["lr"]
                io.cprint(
                    f"Current Learning Rate: {current_lr}, Trigger Times: {triggertimes}, Scheduler: Plateau"
                )
            else:
                scheduler2.step()
                current_lr = scheduler2.optimizer.param_groups[0]["lr"]
                io.cprint(f"Current Learning Rate: {current_lr}, Scheduler: Step")
    return best_loss
                

def test(params, test_set):
    # Define Device & Experiment Name
    device = torch.device("cuda" if params["cuda"] else "cpu")
    exp_name = params["exp_name"]

    # Set Up Model
    if params["model"] == "fusion_unet":  # Fusion Unet
        model = FusionUNet(in_channels=params["in_channels"], num_classes=len(params["classes"]), num_models=params['num_models'], join=params['join']).to(device).cuda()
    elif params["model"] == "unet":  # Unet
        model = (
            UNet(in_channels=params["in_channels"], num_classes=len(params["classes"]))
            .to(device)
            .cuda()
        )
    elif params["model"] == "ensamble_unet": # ensamble unet
        model = (
            EnsambleUNet(in_channels=params["in_channels"], num_classes=len(params["classes"]), join=params['join'])
            .to(device)
            .cuda()
        )
    elif params["model"] == "retain_unet":
        model = RetFuseNet(in_channels=params["in_channels"], num_classes=len(params["classes"]), num_models=params['num_models'], join=params['join']).to(device).cuda()
    else:
        model_name = params["model"]
        raise Exception(f"Model: {model_name} Not Implemented")

    # Run in Parallel
    if params["n_gpus"] > 1:
        model = nn.DataParallel(
            model.cuda(), device_ids=list(range(0, params["n_gpus"]))
        )

    # Load trained model
    model.load_state_dict(torch.load(params["model_path"]))

    # Load test set
    test_loader = DataLoader(
        test_set, batch_size=params["batch_size"], shuffle=False, pin_memory=True
    )

    # Set up testing
    model.eval()

    with torch.no_grad():
        test_r2s = []

        # Testing
        for data, label, filename in tqdm(
            test_loader, desc="Testing Total: ", leave=False, colour="green"
        ):
            data, label = (
                # data.to(device),
                {k: v.to(device) for k, v in data.items()},
                label.to(device).squeeze()
            )

            # Run Model
            output = model(data)
            output = F.softmax(output, dim=1)

            # Calculate R2
            r2_score = R2Score()
            r2 = r2_score(output, label)
            test_r2s.append(r2.item())

            # Write output to file
            for x, file in enumerate(filename):
                raster_folder = params["raster_folder"][0]
                tensor_to_geotiff(
                    output[x],
                    f"checkpoints/{exp_name}/output/test_rasters/{file}",
                    os.path.join(raster_folder, file),
                    label[x],
                )

        # Get Mean R2
        test_r2 = np.mean(test_r2s)
        print(f"Testing R2: {test_r2}")

        # Send Telegram
        token = params["telegram_token"]
        chat_id = params["telegram_id"]
        try:
            send_telegram(f"Testing R2: {test_r2}", token, chat_id)
        except:
            pass
                
                
def test_dgcnn(params, io, test_set):
    # Define device to use (gpu/cpu)
    device = torch.device("cuda" if params["cuda"] else "cpu")

    # Load model
    model = DGCNN(params, len(params["classes"])).to(device)

    # Initiate Data Parallel
    model = nn.DataParallel(model, device_ids=list(range(0, params["n_gpus"])))

    # Load pretrained model
    model.load_state_dict(torch.load(params["model_path"]))

    # Setup for testing
    model = model.eval()
    test_true = []
    test_pred = []
    ids_true = []
    test_loader = DataLoader(
        test_set, batch_size=params["batch_size"], shuffle=False, pin_memory=True, collate_fn=custom_collate_fn
    )

    # Testing

    for data, label, ids in tqdm(
        test_loader, desc="Running DGCNN: ", leave=False, colour="green"
    ):
        try:
            if any(x is None for x in [data, label, ids]):
                print(f"Skipping Batch due to NoneType in data, label or ids")
                continue
            else:
                data, label = (data.to(device), label.to(device).squeeze())
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]

                # Run model
                output = F.softmax(model(data), dim=1)

                # append true/pred
                test_true.append(label.cpu().numpy().round(2))
                test_pred.append(output.detach().cpu().numpy())
                ids_true.append(ids)
        except FileNotFoundError as e:
            print(e)
            continue
            
    # Concatenate true/pred
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    ids_true = np.concatenate(ids_true)
    
    # Calculate R2
    # r2_score = R2Score()
    r2 = sk_r2(test_true.flatten(), test_pred.flatten().round(2))
    
    # Send Telegram
    token = params["telegram_token"]
    chat_id = params["telegram_id"]
    send_telegram(f"R2: {r2}", token, chat_id)
    
    # Create output dataframe
    
    out_df = pd.DataFrame({"plot_id": ids_true, "y_true": test_true.tolist(), "y_pred": test_pred.tolist()})
    
    exp_name = params["exp_name"]
    out_df.to_pickle(f"checkpoints/{exp_name}/output/test_dgcnn.pkl")