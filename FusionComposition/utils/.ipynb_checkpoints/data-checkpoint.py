import os

import laspy
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)+
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    try:
        inFile = laspy.read(pointcloudfile)
    except Exception as e:
        print(f"Failed to read {pointcloudfile}: {e}")
        return (np.zeros((7168, 3)), {}) if get_attributes else np.zeros((7168, 3))
    # Get coordinates (XYZ)
    if len(inFile.x) == 7168:
        coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    else:
        coords = np.zeros((7168, 3))
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)
    
    
def read_raster(rasterfile):
    with rasterio.open(rasterfile) as src:
        src_np = src.read()
        return src_np
    
def raster_augmentations(imgs, target):
    flip_rotation = [
        "flip_horizontal",
        "flip_vertical",
        "rotate_90",
        "rotate_180",
        "rotate_270",
    ]
    brightness_contrast = ["brightness", "contrast"]

    augment_1 = random.choice(flip_rotation)
    augment_2 = random.choice(brightness_contrast)

    imgs = list(imgs.values())

    aug_imgs_tensor = {}
    target = target.numpy()

    if augment_1 == "flip_horizontal":
        aug_imgs = [np.flip(img.numpy(), axis=2) for img in imgs]
        aug_target = np.flip(target, axis=2)
    elif augment_1 == "flip_vertical":
        aug_imgs = [np.flip(img.numpy(), axis=1) for img in imgs]
        aug_target = np.flip(target, axis=1)
    elif augment_1 == "rotate_90":
        aug_imgs = [np.rot90(img.numpy(), k=1, axes=(1, 2)) for img in imgs]
        aug_target = np.rot90(target, k=1, axes=(1, 2))
    elif augment_1 == "rotate_180":
        aug_imgs = [np.rot90(img.numpy(), k=2, axes=(1, 2)) for img in imgs]
        aug_target = np.rot90(target, k=2, axes=(1, 2))
    elif augment_1 == "rotate_270":
        aug_imgs = [np.rot90(img.numpy(), k=3, axes=(1, 2)) for img in imgs]
        aug_target = np.rot90(target, k=3, axes=(1, 2))

    if augment_2 == "brightness":
        brightness_factor = np.random.uniform(0.5, 1.5)
        aug_imgs = [img * brightness_factor for img in aug_imgs]
    elif augment_2 == "contrast":
        contrast_factor = np.random.uniform(0.5, 1.5)
        aug_imgs = [
            (img - np.mean(img, axis=(1, 2), keepdims=True)) * contrast_factor
            + np.mean(img, axis=(1, 2), keepdims=True)
            for img in aug_imgs
        ]

    for i, aug_img in enumerate(aug_imgs):
        aug_img = torch.from_numpy(aug_img).float()
        aug_imgs_tensor[f"img_{i}"] = aug_img

    aug_target = torch.from_numpy(aug_target.copy()).float()
    return aug_imgs_tensor, aug_target
    
class PointCloudsInPickle(Dataset):
    def __init__(self, filepath, pickle, id_column="", class_column=""):
        self.filepath = filepath
        self.pickle = pd.read_pickle(pickle)
        self.id_column = id_column
        self.class_column = class_column
        super().__init__()
        
    def file_exists(self, file_path):
        return os.path.exists(file_path), file_path
    
    def validate_files(self):
        # Prepare file paths to check
        file_paths = [os.path.join(self.filepath, str(ids) + ".laz") for ids in self.pickle[self.id_column]]

        # Use ThreadPoolExecutor to parallelize file existence checks
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_path = {executor.submit(self.file_exists, path): path for path in file_paths}
            
            # Collect paths of files that exist
            valid_files = {future.result()[1] for future in as_completed(future_to_path) if future.result()[0]}
            
        # Filter the dataframe to keep only rows with existing files
        valid_indices = [i for i, path in enumerate(file_paths) if path in valid_files]
        self.pickle = self.pickle.iloc[valid_indices].reset_index(drop=True)
        
    def get_default_values(self):
        coords_placeholder = torch.zeros((7168, 3))
        target_placeholder = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1]).float()
        return coords_placeholder, target_placeholder


    def __len__(self):
        return len(self.pickle)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get filename
        pickle_idx = self.pickle.iloc[idx : idx + 1]
        ids = pickle_idx[self.id_column].item()
        file = os.path.join(self.filepath, str(ids) + ".laz")

        try:
            # Read Point Cloud
            coords = read_las(file, get_attributes=False)

            # Normalize coords to mean
            coords = coords - np.mean(coords, axis=0)

            # Get Target
            target = pickle_idx[self.class_column].item()
            # target = target.replace("[", "")
            # target = target.replace("]", "")
            # target = target.split(",")
            target = [round(i, 2) for i in target[0]] # convert to float

            # Send to tensors
            coords = torch.from_numpy(coords).float()
            target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
            
        except Exception as e:
            print(f"Warning: Failed to read {file}: {e}")
            coords, target = self.get_default_value()        

        return coords, target, ids
    


class RastersInDF(Dataset):
    def __init__(self, rasterfolders, labelfolder, df, augment=False):
        self.rasterfolders = rasterfolders
        self.labelfolder = labelfolder
        self.df = df
        self.augment=augment
        super().__init__()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get filename
        df_idx = self.df.iloc[idx : idx + 1]
        filename = df_idx["Filename"].item()

        # Load target
        label_path = os.path.join(self.labelfolder, filename)
        target = read_raster(label_path).astype("float16")
        target = torch.from_numpy(target).float()

        # Load Rasters
        imgs = {}
        for i, rasterfolder in enumerate(self.rasterfolders):
            raster_path = os.path.join(rasterfolder, filename)
            img = read_raster(raster_path).astype("int16")
            img = torch.from_numpy(img).float()
            imgs[f"img_{i}"] = img
            
        if self.augment:
            imgs, target = raster_augmentations(imgs, target)
            
        return imgs, target, filename