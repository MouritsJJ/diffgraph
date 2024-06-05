import numpy as np
import matplotlib.pyplot as plt

def cosine_schedule(beta_min, beta_max, timesteps):
    # https://openreview.net/forum?id=-NEXDKk8gZ
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

def linear_schedule(beta_min, beta_max, timesteps):
    # https://arxiv.org/abs/2006.11239
    # Computes the linear noise schedule variables: beta, alpha, alpha_bar
    beta = np.linspace(beta_min, beta_max, timesteps)
    alpha = 1. - beta
    alpha_bar = np.cumprod(alpha, axis=0)

    return beta, alpha, alpha_bar


if __name__ == "__main__":
    *_, cosine = cosine_schedule(0, 0.999, 1000)
    beta, alpha, alpha_bar = linear_schedule(1e-4, 0.02, 1000)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(cosine, label="Cosine")
    ax.plot(alpha_bar, label="Linear")
    ax.set_ylabel(r"$\bar{\alpha}_t$", fontsize=12)
    ax.set_xlabel("Timestep (t/T)", fontsize=11)
    plt.title("Comparison of noise accumulation", fontsize=13)
    plt.legend()
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(-0.2, 1.3, 0.2)])
    plt.show()
