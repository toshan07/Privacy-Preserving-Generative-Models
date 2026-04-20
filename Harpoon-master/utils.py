import torch


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    Sigma[0] = 0
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams
