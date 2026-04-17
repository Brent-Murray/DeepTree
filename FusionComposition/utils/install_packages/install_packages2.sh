#!/bin/bash

# Create environment
conda activate gpu_env

conda install -c numba numba

# List of packages to install (anaconda)
packages=("pandas" "seaborn" "scikit-learn" "scikit-image" "cupy")

# Install packages
for package in "${packages[@]}"
do
	conda install -n fusion-dl -c anaconda "$package" -y
done

# List of packages to install (conda-forge)
packages=("geopandas" "matplotlib" "laspy" "lazrs-python" "laszip" "scikit-learn-intelex" "rasterio")

# Install packages
for package in "${packages[@]}"
do
	conda install -n fusion-dl -c conda-forge "$package" -y
done