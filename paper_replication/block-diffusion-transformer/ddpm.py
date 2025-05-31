import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S %p')

class DDPM:
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 64, device = "cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_schedule()
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bar)
    def get_timestep(self, tensor: torch.Tensor, timestep: int, x) -> torch.Tensor:
        return tensor[timestep].reshape(x.shape[0], 1, 1, 1)

    def prepare_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_image(self, x0 :torch.Tensor, timestep:int, epsilon:torch.Tensor=None): # q process
        if epsilon is None:
            epsilon = torch.randn_like(x0) # Actual noise added to the image
        return self.get_timestep(self.sqrt_alpha_bar, timestep, x0) * x0 + self.get_timestep(self.sqrt_one_minus_ab, timestep, x0) * epsilon, epsilon

    def sample(self, model: nn.Module, n):
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(range(1, self.noise_steps), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise
