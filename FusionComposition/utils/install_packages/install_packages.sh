#!/bin/bash

# Create environment
conda create --name fusion-dl -c conda-forge python=3.11
conda activate fusion-dl

# Install Pytorch
conda install pytorch torchvision torchaudio pytorch=cuda=11.7 -c pytorch -c nvidia

# List of packages to install (anaconda)
packages=("pandas" "seaborn" "scikit-learn" "seaborn" "scikit-image")

# Install packages
for package in "${packages[@]}"
do
	conda install -n fusion-dl -c anaconda "$package" -y
done

# List of packages to install (conda-forge)
packages=("geopandas" "matplotlib" "laspy" "lazrs-python" "laszip" "scikit-learn-intelex")

# Install packages
for package in "${packages[@]}"
do
	conda install -n fusion-dl -c conda-forge "$package" -y
done