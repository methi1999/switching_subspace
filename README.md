# Switching Subspace Model for Neural Population Analysis

This repo contains the code for the paper "Switching Subspace Model for Neural Population Analysis".


## Repository Structure

Here's a list of the main files and notebooks in this repository, along with brief descriptions:

### Jupyter Notebooks
- `main.ipynb`: Main notebook for training the SSM and comparing it to other models.
- `dandi_preprocess.ipynb`: Handles data preprocessing steps for neural recordings. Download the data from DANDI and run this notebook to preprocess the data.
- `cnn.ipynb`: Training a CNN on the raw spike data.
- `compare.ipynb`: Code for plotting the results across different models.

### Files
- `requirements.txt`: List of Python dependencies for the project.
- `data_loader.py`: Loading data.
- `early_stopping.py`: Early stopping code for halting training.
- `hyperparam.py`: Code for hyperparameter tuning.
- `metrics.py`: Code for computing metrics such as bits/spike and accuracy.
- `model.py`: Switching Subspace Model.
- `train_test.py`: Training and testing loop for the model.
- `utils.py`: Utility functions for the project.
- `config.yaml`: Configuration file for the project.

### Folders

- `results/`: Directory for storing model outputs and generated figures.
- `utils/`: Directory containing utility functions used across the project.
- `misc/`: Miscellaneous scripts and explorations..
- `img/`: Image files for the paper.
- `pace_scripts/`: Scripts for training on PACE.
- `supervised_decoders/`: Model files for the supervised decoders on top of the latents e.g. MLP/CNN
- `vae/`: Model files for the VAE and GP-VAE

## Usage

[Provide brief instructions on how to use the main scripts or functions]

## Data

[Explain where to find the data used in the paper, or how to use example datasets]

## Results

[Briefly describe the key findings or include links to result figures]

## Citation

If you use this code in your research, please cite our paper:
