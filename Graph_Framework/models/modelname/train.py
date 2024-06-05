from typing import *

def loss_fn(x_0, con_diffusion, cat_diffusion, model) -> Tuple(Any, str):
    """
    The user specified loss function. 

    Parameters:
    ---
    x_0           : The data sample from the dataset
    con_diffusion : The continuous diffusion module used for adding noise
    cat_diffusion : The categorical diffusion module used for adding noise
    model         : The model being trained

    Returns:
    ---
    loss          : The collective loss used for backpropagation
    log           : A string for logging in tqdm progress bar
    """

def val_fn(val_dataset, con_diffusion, cat_diffusion, model, decode_atom, decode_bond, epoch, config) -> None:
    """
    The user specified validation function.

    Parameters::
    ---
    val_dataset : The dataloder for the validation dataset
    con_diffusion : The continuous diffusion module used for adding noise
    cat_diffusion : The categorical diffusion module used for adding noise
    model       : The model being trained
    decode_atom : The dictionary to decode atom features
    decode_bond : The dictionary to decode bond features
    epoch       : The current epoch
    config      : The training dictionary from config file
    """
