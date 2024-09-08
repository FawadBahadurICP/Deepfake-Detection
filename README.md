# DeepFakeDetection

## Overview

This project focuses on detecting deepfake videos using geometric features extracted from facial landmarks. The model is trained on a dataset of real and fake videos, and predictions are made using a trained neural network.

## Directory Structure

- `data/` - Contains the dataset CSV file.
- `images/` - Directory for storing real and fake images (optional).
- `models/` - Contains the trained models and shape predictor file.
- `results/` - Directory for saving results like accuracy plots and confusion matrices.
- `src/` - Source code including feature extraction, model definitions, and utility functions.
- `test.py` - Script to predict if a video is real or fake.
- `train.py` - Script to train the model and save it.
- `README.md` - Project documentation.

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
