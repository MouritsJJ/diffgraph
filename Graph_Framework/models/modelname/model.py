import torch.nn as nn

class ModelName(nn.Module):
    """
    - Class that contains the model to be used when making predictions
    - ModelName.lower() must match name of the folder
    """
    
    def __init__(self, config) -> None:
        """
        Required to accept the config input

        Parameters:
        ---
        config: The model.params dictionary from the yml config file
        """
        super().__init__()

    def forward(self, params):
        """
        The forward function for the model.
        """