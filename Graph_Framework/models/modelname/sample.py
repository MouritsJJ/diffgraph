from typing import *

def sample_batch(n_samples, dataset, model, device) -> Any:
    """
    Function for sampling a batch of noisy data.
    
    Parameters:
    ---
    n_samples   : The number of samples to generate in the batch
    dataset     : The dataset created based on the parsed parameters
    model       : The model used for cleaning the sample
    device      : The device to send the batch onto

    Returns:
    ---
    The sampled batch x_T
    """

def sample_reverse(con_diffusion, cat_diffusion, model, t, x_t) -> Any:
    """
    The function that takes one step of the reverse process.

    Parameters:
    ---
    con_diffusion : The continuous diffusion module used for removing noise
    cat_diffusion : The categorical diffusion module used for removing noise
    model         : The model used for cleaning the sample
    t             : The current timestep in the reverse process
    x_t           : The current data sample to be cleaned

    Returns:
    ---
    The updated sample x_t-1
    """

def sample_mols(x_0, dataset):
    """
    The function that converts the cleaned sample into RDKit molecules

    Parameters:
    ---
    x_0         : The cleaned data sample
    dataset     : The dataset with information for converting data samples to mols

    Returns:
    ---
    A list of RDKit molecules
    """
