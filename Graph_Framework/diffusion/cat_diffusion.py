import torch
import numpy as np
import importlib


class CategoricalDiffusion(torch.nn.Module):
    """
        A class to implement all the diffusion processes using a categorical distribution.
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

        # Precompute log noise schedule
        alpha = np.sqrt(self.noise_schedule.alpha.cpu().numpy())
        log_alpha = np.log(alpha)
        log_alpha_bar = np.cumsum(log_alpha)
        log_one_minus_alpha = np.log(1 - np.exp(log_alpha) + 1e-40)
        log_one_minus_alpha_bar = np.log(1 - np.exp(log_alpha_bar) + 1e-40)

        # Send log noise schedule to device
        self.alpha = torch.from_numpy(alpha).to(device)
        self.log_alpha = torch.from_numpy(log_alpha).to(device)
        self.log_alpha_bar = torch.from_numpy(log_alpha_bar).to(device)
        self.log_one_minus_alpha = torch.from_numpy(log_one_minus_alpha).to(device)
        self.log_one_minus_alpha_bar = torch.from_numpy(log_one_minus_alpha_bar).to(device)

    def index_to_log_one_hot(self, x, K):
        """
            Encode K categories to log one-hot.

            Parameters
            ---
            x   : Categories to encode.
            K   : Number of categories.

            Returns
            ---
            x   : One-hot encoded version of the input object.
        """
        onehot = torch.nn.functional.one_hot(x, K)
        return torch.log(onehot.float().clamp(min=1e-30))
    
    def stable_log_add(self, a, b):
        """
            Stable addition of log values

            Parameters
            ---
            a   : First term of addition.
            b   : Second term of addtion.

            Returns
            ---
            sum : The sum of a and b in log space.
        """
        maximum = torch.max(a, b)
        return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))
    
    def sample_log_categorical(self, x, mask, K):
        """
            Sample from categorical distribution using Gumbel noise.

            Parameters:
            ---
            x       : Probabilities for each cateorgy used for sampling.
            mask    : The mask used to mask the object after sampling.
            K       : Number of categories.

            Returns
            ---
            x_t     : One-hot encoded noisy object.
        """
        # Sample Gumbel noise
        uniform = torch.rand_like(x)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)

        # Category 0 is used as padding. -5 is lower than min(gumbel_noise)
        gumbel_noise[..., 0] = -5.

        # Sample the data and apply mask
        sample = (gumbel_noise + x).argmax(dim=-1)
        for _ in range(len(sample.shape) - len(mask.shape)):
            mask = mask.unsqueeze(-1)
        sample = sample * mask.long()

        return self.index_to_log_one_hot(sample, K)
    
    def q_one_step(self, x, t, K):
        """
            Forward or reverse on step in the diffusion process.
            As the process is symmetrical, the same method work in both directions.

            Parameters
            ---
            x   : The object to process.
            t   : The timestep to process the inpur from.
            K   : The number of categories.

            Returns
            ---
            x_t : The process object.
        """
        # Prepare noise schedule values for timestep t
        alpha = self.log_alpha[t]
        one_minus_alpha = self.log_one_minus_alpha[t]
        for _ in x.shape[1:]:
            alpha = alpha.unsqueeze(-1)
            one_minus_alpha = one_minus_alpha.unsqueeze(-1)

        # Take one diffusion step.
        # x_t = log (alpha_t * x + (1 - alpha_t) / K)
        left = alpha + x
        right = one_minus_alpha - np.log(K)
        x_t = self.stable_log_add(left, right)

        return x_t

    def q_xt_given_x0(self, x_0, t, K):
        """
            Calcualte x at timestep t given x at timestep 0.

            Parameters
            ---
            x_0 : The clean object.
            t   : The timestep to forward the input object to.
            K   : The number of categories.

            Returns
            ---
            x_t : The noisy object.
        """
        # Prepare noise schedule values for timestep t
        alpha_bar = self.log_alpha_bar[t]
        one_minus_alpha_bar = self.log_one_minus_alpha_bar[t]
        for _ in x_0.shape[1:]:
            alpha_bar = alpha_bar.unsqueeze(-1)
            one_minus_alpha_bar = one_minus_alpha_bar.unsqueeze(-1)

        # Add noise
        # log c = log (alpha_bar_t * x_0 + (1 - alpha_bar_t) / K)
        left = alpha_bar + x_0
        right = one_minus_alpha_bar - np.log(K)
        x_t = self.stable_log_add(left, right)

        return x_t

    def q_posterior(self, x_t, x_0, t, K):
        """
            Computes the x_t-1 from x_t and x_0

            Parameters
            ---
            x_t     : The noisy sample for timestep t.
            x_0     : The clean sample.
            t       : Timestep for x_t.
            K       : The number of categories.

            Returns
            ---
            c_tilde : Probabilties for x for timestep t - 1.
        """
        # Get timestep t - 1
        t_minus_1 = t - 1
        # Negative timesteps are not used
        t_minus_1 = t_minus_1.clamp(min=0)

        # Compute c_star
        c_star_left = self.q_one_step(x_t, t, K)
        c_star_right = self.q_xt_given_x0(x_0, t, K)
        c_star = c_star_left + c_star_right

        # Compute c_tilde
        c_tilde = c_star - torch.logsumexp(c_star, dim=-1, keepdim=True)

        return c_tilde

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

    def kl_multinomial(self, x, y):
        """
            KL-divergence for multinomial distribution.

            Parameters
            ---
            x   : The predicted distribution.
            y   : The reference distribution.

            Returns
            ---
            kl  : The kl divergence fitting x to y.
        """
        return (x.exp() * (x - y)).sum(dim=-1)

    @torch.no_grad()
    def forward_process(self, x_0, t, node_mask, K):
        """
            The forward process for the diffusion.

            Parameters
            ---
            x_0         : The object to add noise to.
            t           : The timestep to forward the noise to.
            node_mask   : The mask used for masking the object after adding noise.
            K           : The number of categories.

            Returns
            ---
            x_o         : The one-hot encoded clean object.
            x_t         : The noised object.
        """
        # Log one-hot encode x_0
        x_0 = self.index_to_log_one_hot(x_0, K)

        # Get posterior given x0
        x_t = self.q_xt_given_x0(x_0, t, K)

        # Get new sample with added noise
        x_t = self.sample_log_categorical(x_t, node_mask, K)

        return x_0, x_t
    
    @torch.no_grad()
    def reverse_process(self, x_t, pred_x, t, node_mask, K):
        """
            Remove one step of noise from the input.

            Parameters
            ---
            x_t         : Noisy object to remove noise from.
            pred_x      : Prediction used to remove noise.
            t           : Timestep at which x_t is noised to.
            node_mask   : The mask used for masking the input object after removing noise.
            K           : The number of categories.

            Returns
            ---
            x_t_minus_1 : The object cleaned one noise step.
        """
        # Log one-hot encode x_t
        x_t = self.index_to_log_one_hot(x_t, K)

        # Get posterior based on prediction
        x_t_minus_1 = self.q_posterior(x_t, pred_x, t, K)
        
        # Get new sample from posterior
        x_t_minus_1 = self.sample_log_categorical(x_t_minus_1, node_mask, K)
        
        return x_t_minus_1
    
    def get_loss(self, x_0, x_t, pred_x, t, node_mask, K):
        """
            Compute KL-divergence as loss.

            Parameters
            ---
            x_0         : The original clean sample.
            x_t         : The noisy sample.
            pred_x      : The predicted clean sample.
            t           : The timestep for x_t.
            node_mask   : The mask used to mask the loss.
            K           : The number of categories.

            Returns
            ---
            loss        : The kl divergence as loss.
        """

        # True distribution
        x_t_minus_1 = self.q_posterior(x_t, x_0, t, K)
        # Predicted distribution
        pred_x_t_minus_1 = self.q_posterior(x_t, pred_x, t, K)
        # KL-divergence between True and prediction
        kl = self.kl_multinomial(x_t_minus_1, pred_x_t_minus_1)
        kl = kl * node_mask.float()

        # For t = 0 we use NLL as loss
        pred_nll = -(x_0.exp() * pred_x_t_minus_1).sum(dim=-1)
        pred_nll = pred_nll * node_mask.float()
        t_mask = (t == 0).float()
        for _ in range(len(kl.shape) - len(t_mask.shape)): t_mask = t_mask.unsqueeze(-1)
        kl = t_mask * pred_nll + (1. - t_mask) * kl

        # Compute KL-divergence between x_T and uniform -> regularisation term
        x_T = self.q_xt_given_x0(x_0, t * 0 + self.timesteps - 1, K)
        uniform = -torch.log(K * torch.ones_like(x_T))
        kl_prior = self.kl_multinomial(x_T, uniform)
        kl_prior = kl_prior * node_mask.float()
        
        # Compute scaled and regularised loss
        loss = kl * self.timesteps + kl_prior
        return loss.sum() / (np.log(2) * node_mask.sum().float())
