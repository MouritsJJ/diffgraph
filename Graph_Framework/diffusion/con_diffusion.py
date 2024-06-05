import torch
import importlib

class ContinuousDiffusion(torch.nn.Module):
    """
        A class to implement all the diffusion processes using a continuous distribution.
    """
    def __init__(self, config, device):
        """
            Parameters
            ---
            config  : Dictionary containing the entries: timesteps, noise_schedule, and parameters for the noise schedule.
            device  : CUDA device to store the diffusion class on.
        """
        super().__init__()
        # Precompute noise schedule
        self.timesteps = config["timesteps"]
        noise_class = getattr(importlib.import_module(f"diffusion.noise_schedules"), config["noise_schedule"] + "Schedule")
        self.noise_schedule = noise_class(device, config)
        # MSE for calculating the simple noise loss
        self.loss = torch.nn.MSELoss()

    def sample_timesteps(self, n):
        """
            Sample a batch of timesteps from a uniform distribution.

            Parameters
            ---
            n   : Number of timesteps to sample.

            Returns
            ---
            t   : The sampled timesteps.
        """
        return torch.randint(low=1, high=self.timesteps, size=(n,))
    
    @torch.no_grad()
    def forward_process(self, x_0, t, node_mask):
        """
            The forward process for the diffusion.

            Parameters
            ---
            x_0         : The object to add noise to.
            t           : The timestep to forward the noise to.
            node_mask   : The mask used for masking the object after adding noise.

            Returns
            ---
            x_t         : The noised object.
            noise       : The noise added to the object.
        """
        # Adding noise to input
        alpha_bar_t = self.noise_schedule.alpha_bar[t]
        for _ in x_0.shape[1:]:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * Ɛ
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

        # Sample noise from standard Gaussian
        noise = torch.randn_like(x_0)
        for _ in range(len(x_0.shape) - len(node_mask.shape)):
            node_mask = node_mask.unsqueeze(-1)
        noise = noise * node_mask.float()

        # Apply noise
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        return x_t * node_mask.float(), noise
    
    @torch.no_grad()
    def reverse_process(self, x_t, pred_x, t, node_mask):
        """
            Remove one step of noise from the input.

            Parameters
            ---
            x_t         : Noisy object to remove noise from.
            pred_x      : Prediction used to remove noise.
            t           : Timestep at which x_t is noised to.
            node_mask   : The mask used for masking the input object after removing noise.

            Returns
            ---
            x_t_minus_1 : The object cleaned one noise step.
        """
        # Prepare noise schedule variables for timestep t
        beta = self.noise_schedule.beta[t]
        alpha = self.noise_schedule.alpha[t]
        alpha_bar = self.noise_schedule.alpha_bar[t]

        for _ in x_t.shape[1:]:
            beta = beta.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            alpha_bar = alpha_bar.unsqueeze(-1)

        # Do not include noise in the last step
        noise = torch.zeros_like(x_t) if t[0] == 0 else torch.randn_like(x_t)
        
        # Mask noise
        for _ in range(len(x_t.shape) - len(node_mask.shape)):
            node_mask = node_mask.unsqueeze(-1)
        noise = noise * node_mask.float()

        # Remove noise
        # x_t-1 = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * pred_noise) + sqrt(beta_t) * noise
        x_t_minus_1 = 1 / torch.sqrt(alpha) * (x_t - (beta / (torch.sqrt(1 - alpha_bar))) * pred_x) + torch.sqrt(beta) * noise

        return x_t_minus_1

    def get_loss(self, x_0, pred_x):
        """
            Wrapper for MSE loss on added noise.
        """
        return self.loss(x_0, pred_x)
