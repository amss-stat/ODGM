"""Data loading and preprocessing for the HIV-1 drug-resistance analysis."""

import os

import numpy as np
import pandas as pd
import pyreadr
import torch
from sklearn.preprocessing import StandardScaler


def load_hiv_data(data_folder, dataname, device):
    """Load and preprocess one HIV-1 protease-inhibitor data set.

    Parameters
    ----------
    data_folder : str
        Directory containing ``X{dataname}.csv`` and ``y{dataname}.RData``.
    dataname : str
        Drug abbreviation, for example ``"LPV"``.
    device : torch.device
        Device used for the returned tensors.

    Returns
    -------
    x_sample : torch.Tensor
        Column-standardised mutation design matrix.
    y_sample : torch.Tensor
        Standardised log fold-change response vector.
    """
    x_filename = os.path.join(data_folder, f"X{dataname}.csv")
    y_filename = os.path.join(data_folder, f"y{dataname}.RData")

    x_sample = pd.read_csv(x_filename)
    x_sample = x_sample.drop(columns=x_sample.columns[0]).astype(np.float32)
    x_numpy = x_sample.to_numpy()

    nonzero_columns = ~np.all(x_numpy == 0, axis=0)
    x_processed = x_numpy[:, nonzero_columns]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_processed)
    x_sample = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    y_data = pyreadr.read_r(y_filename)
    key = next(iter(y_data))
    vector_data = y_data[key].to_numpy().reshape(-1)
    y_sample = torch.tensor(vector_data, dtype=torch.float32, device=device)
    y_sample = (y_sample - y_sample.mean()) / y_sample.std()

    return x_sample, y_sample
