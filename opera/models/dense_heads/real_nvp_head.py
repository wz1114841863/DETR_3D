import torch
import torch.nn as nn
import numpy as np
from ..builder import HEADS


def nets():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))


class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())

        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm

        if self.bias:
            y = y + self.linear.bias
        return y


@HEADS.register_module()
class RealNVP(nn.Module):
    def __init__(self, 
                    nets=nets(), 
                    nett=nett()):
        super(RealNVP, self).__init__()
        self.mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.register_buffer('mask', self.mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(self.mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(self.mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        # x.shape: [544, 2]
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z  # [544, 2]
            s = self.s[i](z_) * (1 - self.mask[i])  # [544, 2]
            t = self.t[i](z_) * (1 - self.mask[i])  # [544, 2]
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_  # [544, 2]
            log_det_J -= s.sum(dim=1)  # [544, ]
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device 
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)
