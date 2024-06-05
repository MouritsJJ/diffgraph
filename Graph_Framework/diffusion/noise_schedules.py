import torch
import numpy as np

class CosineSchedule:
    """
        Class implementing the cosine noise schedule.
        https://openreview.net/forum?id=-NEXDKk8gZ
    """
    def __init__(self, device, config):
        """
            Parameters
            ---
            device  : CUDA device to place the precomputed values on.
            config  : Dictionary containing the entries: timesteps, beta_min, beta_max
        """
        timesteps = config["timesteps"]
        beta_min = config["beta_min"]
        beta_max = config["beta_max"]
        # Compute noise schedule values
        beta, alpha, alpha_bar = self.cosine_schedule(beta_min, beta_max, timesteps)
        # Send values to device
        self.beta = torch.from_numpy(beta).float().to(device)
        self.alpha = torch.from_numpy(alpha).float().to(device)
        self.alpha_bar = torch.from_numpy(alpha_bar).float().to(device)

    def cosine_schedule(self, beta_min, beta_max, timesteps):
        """
            Precomputes the noise schedule values: beta, alpha, alpha_bar.

            Parameters
            ---
            beta_min    : The minimum beta value.
            beta_max    : The maximum beta value.
            timesteps   : The number of timesteps in the diffusion process.

            Returns
            ---
            beta        : Beta values for the noise schedule.
            alpha       : Alpha values for the noise schedule.
            alpha_bar   : Alpha_bar values for the noise schedule.
        """
        # s parameter proposed in the paper
        s = 0.008
        # Initialise all t values in a list
        t = np.arange(0, timesteps + 1)

        # f(t) = cos(((t / T) + 1) / (1 + s) * 0.5 * pi)^2
        f_t = np.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        # alpha_tilde = f(t) / f(0)
        alpha_tilde = f_t / f_t[0]
        # beta = 1 - (alpha_tilde_t / alpha_tilde_{t-1})
        beta = 1 - (alpha_tilde[1:] / alpha_tilde[:-1])
        # Clip betas to predefined range to avoid too large and small values
        beta = np.clip(beta, a_min=beta_min, a_max=beta_max)

        # Compute alpha and alpha_bar based on formulas
        alpha = 1. - beta
        alpha_bar = np.cumprod(alpha, axis=0)

        return beta, alpha, alpha_bar
    
class LinearSchedule():
    """
        Class implementing the linear noise schedule.
        https://arxiv.org/abs/2006.11239
    """
    def __init__(self, device, config):
        """
            Parameters
            ---
            device  : CUDA device to place the precomputed values on.
            config  : Dictionary containing the entries: timesteps, beta_min, beta_max
        """
        timesteps = config["timesteps"]
        beta_min = config["beta_min"]
        beta_max = config["beta_max"]
        # Compute noise schedule values
        beta, alpha, alpha_bar = self.linear_schedule(beta_min, beta_max, timesteps)
        # Send values to device
        self.beta = torch.from_numpy(beta).float().to(device)
        self.alpha = torch.from_numpy(alpha).float().to(device)
        self.alpha_bar = torch.from_numpy(alpha_bar).float().to(device)

    def linear_schedule(self, beta_min, beta_max, timesteps):
        """
            Precomputes the noise schedule values: beta, alpha, alpha_bar.

            Parameters
            ---
            beta_min    : The minimum beta value.
            beta_max    : The maximum beta value.
            timesteps   : The number of timesteps in the diffusion process.

            Returns
            ---
            beta        : Beta values for the noise schedule.
            alpha       : Alpha values for the noise schedule.
            alpha_bar   : Alpha_bar values for the noise schedule.
        """
        # Computes the linear noise schedule variables: beta, alpha, alpha_bar
        beta = np.linspace(beta_min, beta_max, timesteps)
        alpha = 1. - beta
        alpha_bar = np.cumprod(alpha, axis=0)

        return beta, alpha, alpha_bar