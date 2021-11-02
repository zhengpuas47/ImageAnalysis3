import numpy as np
import math

import torch
import torch.multiprocessing as multiprocessing
import torch.nn.functional as F


class Kernel3D(torch.nn.Module):
    def __init__(self, shape, device):
        super(Kernel3D, self).__init__()

        self.shape = shape
        self.device = device

    def _normalized_basis(self, mu, sigma, num_dim):
        _size = self.shape[num_dim]
        _mu = mu[:, num_dim, None]
        _sigma = sigma[:, num_dim, None]

        return (
            1 / (2 * math.pi * _sigma ** 2) ** (1 / 2) *
            torch.exp(- 1 / 2 * ((_mu - torch.arange(_size, dtype=dtype, device=self.device)) / _sigma) ** 2)
        )

    def forward(self, x, mu, sigma):
        x = torch.einsum('nz,nt->nzt', [self._normalized_basis(mu, sigma, num_dim=2), x])
        x = torch.einsum('ny,nzt->nyzt', [self._normalized_basis(mu, sigma,num_dim=1), x])
        x = torch.einsum('nx,nyzt->xyzt', [self._normalized_basis(mu, sigma,num_dim=0), x])

        return x