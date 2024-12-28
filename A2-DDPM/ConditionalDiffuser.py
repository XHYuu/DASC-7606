import torch.nn as nn
import torch
from torch import Tensor
import math
from ConditionalUnet import ConditionalUnet
from tqdm import tqdm


class ConditionalDiffuser(nn.Module):
    def __init__(self, image_size, in_channels, number_class=10, number_embedding_dim=128, time_embedding_dim=256,
                 timesteps=1000, base_dim=32, dim_mults=[1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        # generate suitable beta to add noice in image. use cos or linear methods
        betas = self._cosine_variance_schedule(timesteps)  # or self._linear_variance_schedule(timesteps)
        alphas = 1. - betas
        # obtain the cumprod from a_1 to a_t
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        self.model = ConditionalUnet(timesteps, time_embedding_dim, number_class, number_embedding_dim,
                                     in_channels, in_channels, base_dim, dim_mults)

    def forward(self, x, noise, class_number):
        # in order to train the model to produce true noise
        # x:NCHW
        # random sampling the timesteps to train the model
        # choose the x_t to train the noise model
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)  # add noise from x_0 to x_t
        pred_noise = self.model(x_t, t, class_number)  # predict noise variable epsilon

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples: int, number: int, device="cuda") -> Tensor:
        # produce init noise from normal distribution
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        number = torch.tensor([i for i in range(number)] * 4, device=device)
        # from xT to x0
        for i in range(self.timesteps - 1, -1, -1):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t = self._reverse_diffusion_with_clip(x_t, t, number, noise)

        x_t = (x_t + 1.) / 2.  # [-1,1] to [0,1]

        return x_t

    def _cosine_variance_schedule(self, timesteps: int, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def _linear_variance_schedule(self, timesteps: int, beta_1=1e-4, beta_T=0.02):
        '''
            generate cosine variance schedule
            reference: the DDPM paper https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
            You might compare the model performance of linear and cosine variance schedules.
        '''
        # raise NotImplementedError
        # ---------- **** ---------- #
        # YOUR CODE HERE
        # a queue that beta increase from beta_1 to beta_T
        betas = torch.linspace(beta_1, beta_T, timesteps, dtype=torch.float32)
        return betas
        # ---------- **** ---------- #

    def _forward_diffusion(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        '''
            forward diffusion process
            hint: calculate x_t given x_0, t, noise
            please note that alpha related tensors are registered as buffers in __init__, you can use gather method to get the values
            reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process
        '''
        # raise NotImplementedError
        # ---------- **** ---------- #
        # YOUR CODE HERE
        x_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t) \
                  .reshape(x_0.shape[0], 1, 1, 1) * noise + \
              self.sqrt_alphas_cumprod.gather(0, t) \
                  .reshape(x_0.shape[0], 1, 1, 1) * x_0
        return x_t
        # ---------- **** ---------- #

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t: Tensor, t: Tensor, n: Tensor, noise: Tensor) -> Tensor:
        '''
            reverse diffusion process with clipping
            hint: with clip: pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
                  without clip: pred_noise -> pred_mean and pred_std
                  you may compare the model performance with and without clipping
        '''
        pred = self.model(x_t, t, n)  # predict noise
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        # compute x_0
        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        x_0_pred.clamp_(-1., 1.)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            # conpute p(x_{t-1}|x_t,x_0)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred + \
                   ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0
        # base (0,1) noise normal distribution to re-parameterize
        return mean + std * noise
