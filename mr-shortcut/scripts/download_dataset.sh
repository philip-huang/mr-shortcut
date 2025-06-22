#!/bin/bash

# Install gdown if not already installed
pip install gdown
# Download the dataset from Google Drive
cd ../
gdown 124UENhk04nAFtKsALYTuaoMwy9W30rac
# Extract the dataset to the outputs directory
tar xvzf trajectories.tar.gz 