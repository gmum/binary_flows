"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
import math
from typing import Optional
from scipy.special import gammaln
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
from layers.made import MADE
import voronoi
import argmax_utils


def single_net(M, ks, pad, ch=1):
    net = nn.Sequential(nn.Conv2d(ch, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, 1, kernel_size=ks, padding=pad, stride=1),
                        nn.Sigmoid(),)
    return net


def single_net_conditional(M, ks, pad, conditional_classes, ch=1):
    net = nn.Sequential(nn.Conv2d(ch+conditional_classes, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                        nn.GELU(),
                        nn.Conv2d(M, 1, kernel_size=ks, padding=pad, stride=1),
                        nn.Sigmoid(),)
    return net


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logpdf(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logpdf(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)], device=z.device)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def simplex_uniform_dequantize(x, K):
    # sample uniformly from the simplex
    samples = sample_simplex_uniform(K, shape=x.shape).to(x.device)

    # make index x the top probability
    argmax = torch.argmax(samples, dim=-1).unsqueeze(-1)
    x_idx = x.unsqueeze(-1)
    indices = (
        torch.arange(K + 1)
        .reshape(*([1] * x.ndim), K + 1)
        .expand(samples.shape)
        .to(x.device)
    )
    indices = torch.scatter(indices, -1, x_idx, argmax)
    indices = torch.scatter(indices, -1, argmax, x_idx)

    samples = torch.gather(samples, -1, indices)
    logp = torch.as_tensor(
        simplex_uniform_dequantize_logpdf(K), dtype=torch.float32, device=x.device
    ).expand(x.shape)
    return samples, logp


def simplex_uniform_dequantize_logpdf(K):
    return simplex_uniform_logpdf(K) + math.log(K + 1)


def logsimplex_uniform_dequantize(x, K):
    samples, logp = simplex_uniform_dequantize(x, K)
    logits = torch.log(samples + 1e-10)
    logp = logp + logits[..., :-1].sum(-1)
    return logits, logp


def sample_simplex_uniform(K, shape=(), dtype=torch.float32):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype))[0]
    x = torch.cat(
        [torch.zeros(*shape, 1, dtype=dtype), x, torch.ones(*shape, 1, dtype=dtype)],
        dim=-1,
    )
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


def simplex_uniform_logpdf(K):
    return gammaln(K + 1)


def quantize(samples):
    return torch.argmax(samples, -1)


def _logaddexp(a, b):
    m = torch.max(a, b).detach()
    return torch.log(torch.exp(a - m) + torch.exp(b - m)) + m


class StepSigmoid(nn.Module):
    def __init__(self, temp=1.e7):
        super(StepSigmoid, self).__init__()

        self.temp = temp

    def forward(self, x):
        return torch.sigmoid(x * self.temp)


class SignSTE(nn.Module):
    def __init__(self, threshold=1.):
        super(SignSTE, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        y = F.hardtanh(x, -self.threshold, self.threshold)
        return y + torch.sign(x + 1.e-7).detach() - y.detach()


class StepSTE(nn.Module):
    def __init__(self, threshold=1.):
        super(StepSTE, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        y = F.hardtanh(x, -self.threshold, self.threshold)
        return y + 0.5 * (torch.sign(x + 1.e-7).detach() + 1.) - y.detach()


def create_binary_flows_nets(bits=8, architecture='cnn', M=32, ks=3, pad=1, linear_codes=False, conditional=False, conditional_classes=8):
    if architecture == 'cnn':

        net_a = lambda: nn.Sequential(nn.Conv2d(bits, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, bits, kernel_size=1, padding=0, stride=1),
                                      StepSTE(threshold=0.5))

        net_b = lambda: nn.Sequential(nn.Conv2d(bits, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                      nn.GELU(),
                                      nn.Conv2d(M, bits, kernel_size=1, padding=0, stride=1),
                                      StepSTE(threshold=0.5))

        net_a_no_linear_codes = lambda: nn.Sequential(nn.Conv2d(bits//2, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                      StepSTE(threshold=0.5))

        net_b_no_linear_codes = lambda: nn.Sequential(nn.Conv2d(bits//2, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                      nn.GELU(),
                                                      nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                      StepSTE(threshold=0.5))

        #TODO!!! - create conditional_net_a and conditional_net_b
        conditional_net_a = lambda: nn.Sequential(nn.Conv2d((bits//2) + conditional_classes, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                  StepSTE(threshold=0.5))

        conditional_net_b = lambda: nn.Sequential(nn.Conv2d((bits//2) + conditional_classes, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                  nn.GELU(),
                                                  nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                  StepSTE(threshold=0.5))

        conditional_net_a_no_linear_codes = lambda: nn.Sequential(nn.Conv2d(bits+conditional_classes, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                                  StepSTE(threshold=0.5))

        conditional_net_b_no_linear_codes = lambda: nn.Sequential(nn.Conv2d(8+conditional_classes, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, M, kernel_size=ks, padding=pad, stride=1),
                                                                  nn.GELU(),
                                                                  nn.Conv2d(M, bits//2, kernel_size=1, padding=0, stride=1),
                                                                  StepSTE(threshold=0.5))

        if linear_codes:
            if conditional == True:
                nets = [conditional_net_a, conditional_net_b]
            else:
                nets = [net_a, net_b]
        else:
            if conditional == True:
                nets = [conditional_net_a_no_linear_codes, conditional_net_b_no_linear_codes]
            else:
                nets = [net_a_no_linear_codes, net_b_no_linear_codes]

    return nets


class BaseGaussianDistribution(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, z, **kwargs):
        return standard_normal_logpdf(z).sum(1, keepdim=True)

    def sample(self, num_samples, device, **kwargs):
        z = torch.randn(num_samples, self.d, device=device)
        log_p = self(z)
        return z, log_p


class ConditionalGaussianDistribution(nn.Module):
    def __init__(self, d, cond_embed_dim, actfn):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(cond_embed_dim, 1024), actfns[actfn](), nn.Linear(1024, d * 2)
        )

    def _get_params(self, context):
        out = self.net(context)
        mean, log_std = torch.split(out, (self.d, self.d), dim=1)
        return mean, log_std

    def forward(self, z, context, **kwargs):
        mean, log_std = self._get_params(context)
        return normal_logpdf(z, mean, log_std).sum(1, keepdim=True)

    def sample(self, num_samples, context, device="cpu", **kwargs):
        mean, log_std = self._get_params(context)
        z = torch.randn(num_samples, self.d, device=device)
        z = z * log_std.exp() + mean
        log_p = normal_logpdf(z, mean, log_std).sum(1, keepdim=True)
        return z, log_p


class SemiAutoregressiveBernoulliDistribution(nn.Module):
    def __init__(self, img_shape, M=32, ks=3, pad=1, conditional=False, conditional_classes=8, linear_codes=False):
        super().__init__()
        self.img_shape = img_shape
        self.M = M
        self.ks = ks
        self.pad = pad
        self.conditional = conditional
        self.conditional_classes = conditional_classes
        self.linear_codes = linear_codes
        linear_codes_size = 8 if self.linear_codes else 0
        if self.conditional:
            self.net_base = torch.nn.ModuleList([single_net_conditional(self.M, self.ks, self.pad, i+1) for i in range(self.conditional_classes - 1 + linear_codes_size)])
        else:
            self.net_base = torch.nn.ModuleList([single_net(self.M, self.ks, self.pad, i+1) for i in range(self.conditional_classes - 1 + linear_codes_size)])
        # self.net = nn.Sequential(
        #     nn.Linear(cond_embed_dim, 1024), actfns[actfn](), nn.Linear(1024, d * 2)
        # )
        means_shape = [1, *self.img_shape]
        means_shape[1] = 1
        lin_mean = torch.zeros(means_shape)
        self.lin_mean = nn.Parameter(lin_mean)

    def _log_base(self, x, context=None):
        # TODO - check!!!
        # calculate probs:
        probs = torch.sigmoid(self.lin_mean)
        ones = [1] * (len(self.img_shape))
        probs = probs.repeat(x.shape[0], *ones)

        if context is not None:
            for i in range(len(self.net_base)):
                partial_x = x[:, 0:i+1]
                extended_context = context.repeat(1, 1, 2).reshape((context.shape[0], 4, 1, 2))
                partial_x_with_context = torch.cat((partial_x, extended_context), 1)
                p_i = self.net_base[i](partial_x_with_context)
                probs = torch.cat((probs, p_i), 1)
        else:
            for i in range(len(self.net_base)):
                p_i = self.net_base[i](x[:, 0:i+1])
                probs = torch.cat((probs, p_i), 1)

        probs = torch.clamp(probs, 1.e-7, 1. - 1.e-7)

        log_p = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
        sum_over = [i + 1 for i in range(len(x.shape) - 1)]  # all indices but the batch dimension
        return log_p.sum(sum_over)

    def forward(self, z, context=None, **kwargs):
        # TODO - check if - or +
        # -self._log_base(z, context).sum()
        return self._log_base(z, context).sum()

    def sample(self, num_samples, context=None, device="cpu", **kwargs):
        # FIRST PLANE
        # Calculating probabilities
        probs = torch.sigmoid(self.lin_mean).to(device)
        ones = [1] * (len(self.img_shape))
        probs = probs.repeat(num_samples, *ones)
        # Sampling from the base distribution
        z = torch.bernoulli(probs).to(device)

        if context is not None:
            extended_context = context.repeat(1, 1, 2).reshape((context.shape[0], 4, 1, 2))
            for i in range(len(self.net_base)):
                z_with_context = torch.cat((z, extended_context), 1)
                z_i = torch.bernoulli(self.net_base[i](z_with_context))
                z = torch.cat((z, z_i), 1)
        else:
            # REMAINING PLANES
            for i in range(len(self.net_base)):
                z_i = torch.bernoulli(self.net_base[i](z))
                z = torch.cat((z, z_i), 1)
        # TODO - check!!!
        log_p = z * torch.log(probs) + (1. - z) * torch.log(1. - probs)
        return z, log_p


class ResampledGaussianDistribution(nn.Module):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network,
    see arXiv 1810.11428

    Code based on: https://github.com/VincentStimper/resampled-base-flows
    """

    def __init__(self, d, net_a, T, eps, trainable=True):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        """
        super().__init__()
        self.d = d
        self.net_a = net_a
        self.T = T
        self.eps = eps
        self.register_buffer("Z", torch.tensor(-1.0))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def a(self, inputs, **kwargs):
        return torch.sigmoid(self.net_a(inputs))

    def sample(self, num_samples, device, **kwargs):
        t = 0
        eps = torch.zeros(num_samples, self.d, dtype=self.loc.dtype, device=device)
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T):
            eps_ = torch.randn(
                (num_samples, self.d), dtype=self.loc.dtype, device=device
            )
            acc = self.a(eps_)
            if self.training or self.Z < 0.0:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    eps[s, :] = eps_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p_gauss = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(self.log_scale, 1)
            - torch.sum(0.5 * torch.pow(eps, 2), 1)
        )
        acc = self.a(eps)
        if self.training or self.Z < 0.0:
            eps_ = torch.randn(
                (num_samples, self.d), dtype=self.loc.dtype, device=device
            )
            Z_batch = torch.mean(self.a(eps_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.0:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return z, log_p.reshape(-1, 1)

    def forward(self, z):
        eps = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = (
            -0.5 * self.d * np.log(2 * np.pi)
            - torch.sum(self.log_scale, 1)
            - torch.sum(0.5 * torch.pow(eps, 2), 1)
        )
        acc = self.a(eps)
        if self.training or self.Z < 0.0:
            eps_ = torch.randn_like(z)
            Z_batch = torch.mean(self.a(eps_))
            if self.Z < 0.0:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return log_p.reshape(-1, 1)

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.0
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.d), dtype=dtype, device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches


class MixtureDistribution(nn.Module):
    def __init__(self, *distributions):
        super(MixtureDistribution, self).__init__()
        self.distributions = nn.ModuleList(distributions)
        self.mix_logits = nn.Parameter(torch.zeros(len(self.distributions)))

    def forward(self, z):
        logpdfs = []
        for i in range(len(self.distributions)):
            logpdfs.append(self.distributions[i](z))
        logpdfs = torch.stack(logpdfs, dim=0)

        mix_logprobs = torch.log_softmax(self.mix_logits, 0).reshape(
            -1, *([1] * (logpdfs.ndim - 1))
        )
        return torch.logsumexp(logpdfs + mix_logprobs, dim=0)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        num_samples = int(np.prod(shape))
        samples = []
        for i in range(len(self.distributions)):
            samples.append(self.distributions[i].sample(num_samples))
        samples = torch.stack(samples, dim=0)
        idxs = torch.multinomial(
            torch.softmax(self.mix_logits, 0), num_samples, replacement=True
        )
        samples = samples[idxs, torch.arange(num_samples)].reshape(
            *shape, *samples.shape[2:]
        )
        return samples


class FlowDistrbution(nn.Module):
    def __init__(self, dim, flow, base):
        super(FlowDistrbution, self).__init__()
        self.dim = dim
        self.flow = flow
        self.base = base

    def forward(self, x):
        zeros = torch.zeros(x.shape[0], 1, device=x.device)
        z, logdiff = self.flow(x, logp=zeros)
        logpz = self.base(z).reshape(z.shape[0], -1).sum(1, keepdim=True)
        return logpz - logdiff

    def sample(self, num_samples):
        device = next(self.parameters()).device
        z0, logpz0 = self.base.sample(num_samples, device=device)
        return self.flow.inverse(z0, logp=logpz0)


class Reshape(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.D = D
        self.K = K

    def forward(self, x, logp=None, **kwargs):
        y = x.reshape(*x.shape[:-2], self.D * self.K)
        if logp is None:
            return y
        else:
            return y, logp

    def inverse(self, y, logp=None, **kwargs):
        x = y.reshape(*y.shape[:-1], self.D, self.K)
        if logp is None:
            return x
        else:
            return x, logp


class BinaryFlow(nn.Module):
    def __init__(self, nets, args):
        super(BinaryFlow, self).__init__()

        # parameterization
        self.t_a = torch.nn.ModuleList([nets[0]() for _ in range(args.num_flows)])
        self.t_b = torch.nn.ModuleList([nets[1]() for _ in range(args.num_flows)])

        means_shape = [1, *args.img_shape]
        means_shape[1] = args.bits * 2
        lin_mean = torch.ones(means_shape) * 0.5
        lin_mean[:, 0:args.bits] = -2.
        lin_mean[:, args.bits:] = 2.
        self.lin_mean = nn.Parameter(lin_mean)

        # hyperparameters
        self.args = args

        self.sampling = True
        self.reconstruction = False

    @staticmethod
    def xor(x, y):
        return x + y - 2 * x * y

    @staticmethod
    def permute(x):
        return x.flip(1)

    # Flow-related functions
    def coupling(self, x, index, forward=True):
        # divide into two parts
        (xa, xb) = torch.chunk(x, 2, dim=1)

        # forward
        if forward:
            ya = self.xor(xa, self.t_a[index](xb))
            yb = self.xor(xb, self.t_b[index](ya))
        # inverse
        else:
            yb = self.xor(xb, self.t_b[index](xa))
            ya = self.xor(xa, self.t_a[index](yb))

        return torch.cat((ya, yb), 1)

    def f(self, x):
        z = x
        for i in range(self.args.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)
        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.args.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)
        return x

    # The base distribution
    def log_base(self, x):
        probs = torch.sigmoid(self.lin_mean)
        log_p = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
        sum_over = [i + 1 for i in range(len(x.shape) - 1)]  # all indices but the batch dimension
        return log_p.sum(sum_over)

    #TODO - add sampling with condition
    def sample(self, batch_size=64, device='cpu'):
        # Calculating probabilities
        probs = torch.sigmoid(self.lin_mean).to(device)
        ones = [1] * (len(self.args.img_shape))
        probs = probs.repeat(batch_size, *ones)
        # Sampling from the base distribution
        z = torch.bernoulli(probs).to(device)
        # Inverting the flow
        x = self.f_inv(z).to(device)
        return x

    # FORWARD
    def forward(self, x, y=None, reduction='avg'):
        z = self.f(x)
        if reduction == 'sum':
            return -self.log_base(z).sum()
        else:
            return -self.log_base(z).mean()


class BinaryFlowGroupARM(nn.Module):
    def __init__(self, base_distribution, linear_codes=False, bits=8, architecture="cnn", num_flows=8, M=32, ks=3, pad=1, conditional=False, conditional_classes=8, log=None):
        super(BinaryFlowGroupARM, self).__init__()

        self.architecture = architecture
        self.num_flows = num_flows
        self.bits = bits
        self.linear_codes = linear_codes
        self.M = M
        self.ks = ks
        self.pad = pad
        self.conditional = conditional
        self.conditional_classes = conditional_classes

        nets = create_binary_flows_nets(bits=self.bits, architecture=self.architecture, M=self.M, ks=self.ks, pad=self.pad, linear_codes=self.linear_codes, conditional=self.conditional, conditional_classes=self.conditional_classes)

        # parameterization
        self.t_a = torch.nn.ModuleList([nets[0]() for _ in range(self.num_flows)])
        self.t_b = torch.nn.ModuleList([nets[1]() for _ in range(self.num_flows)])

        # self.net_base = net_base
        self.base_distribution = base_distribution

        self.log = log

    @staticmethod
    def xor(x, y):
        return x + y - 2 * x * y

    @staticmethod
    def permute(x):
        return x.flip(1)

    # Flow-related functions
    def coupling(self, x, index, forward=True, context=None):
        # divide into two parts
        (xa, xb) = torch.chunk(x, 2, dim=1)

        if context is not None:
            extended_context = context.repeat(1, 1, 2).reshape((context.shape[0], 4, 1, 2))
            context = extended_context
            # forward
            if forward:
                xb_with_context = torch.cat((xb, context), 1)
                # ya = self.xor(xa, self.t_a[index](xb, context))
                ya = self.xor(xa, self.t_a[index](xb_with_context))
                ya_with_context = torch.cat((ya, context), 1)
                # yb = self.xor(xb, self.t_b[index](ya, context))
                yb = self.xor(xb, self.t_b[index](ya_with_context))
            # inverse
            else:
                xa_with_context = torch.cat((xa, context), 1)
                # yb = self.xor(xb, self.t_b[index](xa, context))
                yb = self.xor(xb, self.t_b[index](xa_with_context))
                yb_with_context = torch.cat((yb, context), 1)
                # ya = self.xor(xa, self.t_a[index](yb, context))
                ya = self.xor(xa, self.t_a[index](yb_with_context))
        else:
            # forward
            if forward:
                ya = self.xor(xa, self.t_a[index](xb))
                yb = self.xor(xb, self.t_b[index](ya))
            # inverse
            else:
                yb = self.xor(xb, self.t_b[index](xa))
                ya = self.xor(xa, self.t_a[index](yb))

        return torch.cat((ya, yb), 1)

    def f(self, x, context=None):
        z = x
        for i in range(self.num_flows):
            #TODO - conditional version
            z = self.coupling(z, i, forward=True, context=context)
            z = self.permute(z)
        return z

    def f_inv(self, z, context=None):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            #TODO - conditional version
            x = self.coupling(x, i, forward=False, context=context)
        return x

    # The base distribution
    def log_base(self, x, context=None):
        # return log_p.sum(sum_over)
        return self.base_distribution._log_base(x, context=context)

    def sample(self, batch_size=64, device='cpu', context=None):
        z, log_p = self.base_distribution.sample(batch_size, context=context, device=device)
        x = self.f_inv(z, context=context).to(device)
        return x, log_p

    # FORWARD
    def forward(self, x, transforms, context=None):
        x = torch.stack([transforms(y) for y in x])
        z = self.f(x, context)
        return self.log_base(z, context)


    # def forward(self, x, y=None, context=None, reduction='avg'):
    #     #TODO - conditional version
    #     z = self.f(x, context)
    #     #TODO - check!!!
    #     if reduction == 'sum':
    #         return -self.base_distribution._log_base(z, context).sum()
    #     else:
    #         return -self.base_distribution._log_base(z, context).mean()



class SimplexFlowDistribution(nn.Module):
    def __init__(
        self,
        num_discrete_variables,
        embedding_dim,
        num_blocks,
        hdims,
        base,
        actfn="swish",
        arch="mlp",
        num_transformer_layers=2,
        transformer_d_model=512,
        transformer_dropout=0.0,
        block_transform="affine",
        num_mixtures=32,
        flow_type="coupling",
    ):
        super().__init__()

        self.log_softmax = layers.InvertibleLogSoftmax()
        self.logits_to_probs = layers.LogitsToProbabilities()
        self.reshape = Reshape(num_discrete_variables, embedding_dim)

        self.arch = arch
        if arch == "mlp":
            actnorm_dim = num_discrete_variables * embedding_dim
        elif arch != "mlp":
            actnorm_dim = embedding_dim
        else:
            raise ValueError(f"Unknown arch {self.arch}")

        block_fn = invblock if flow_type == "coupling" else autoreg_invblock

        modules = []
        for i in range(num_blocks):
            modules.append(layers.ActNorm1d(actnorm_dim))
            modules.append(
                block_fn(
                    i,
                    num_discrete_variables,
                    embedding_dim,
                    hdims=hdims,
                    transformer_d_model=transformer_d_model,
                    transformer_dropout=transformer_dropout,
                    num_transformer_layers=num_transformer_layers,
                    actfn=actfn,
                    arch=arch,
                    block_transform=block_transform,
                    num_mixtures=num_mixtures,
                )
            )

        self.flow = layers.SequentialFlow(modules)
        self.base = base

    def forward(self, p, logits=True):
        z = p
        logdiff = torch.zeros(p.shape[0], 1, device=p.device)
        if not logits:
            z, logdiff = self.logits_to_probs.inverse(z, logp=logdiff)
        z, logdiff = self.log_softmax.inverse(z, logp=logdiff)

        if self.arch == "mlp":
            z, logdiff = self.reshape(z, logp=logdiff)
            z, logdiff = self.flow(z, logp=logdiff)
        else:
            z, logdiff = self.flow(z, logp=logdiff)
            z, logdiff = self.reshape(z, logp=logdiff)

        logpz = self.base(z).reshape(z.shape[0], -1).sum(1, keepdim=True)
        return logpz - logdiff

    def sample(self, num_samples, logits=True, return_logpdf=False):
        device = next(self.parameters()).device
        z, logpz = self.base.sample(num_samples, device=device)

        if self.arch == "mlp":
            z, logpz = self.flow.inverse(z, logp=logpz)
            z, logpz = self.reshape.inverse(z, logp=logpz)
        else:
            z, logpz = self.reshape.inverse(z, logp=logpz)
            z, logpz = self.flow.inverse(z, logp=logpz)

        z, logpz = self.log_softmax(z, logp=logpz)

        if not logits:
            z, logpz = self.logits_to_probs(z, logp=logpz)

        if return_logpdf:
            return z, logpz
        else:
            return z


class ConditionalSimplexFlowDistribution(nn.Module):
    def __init__(
        self,
        num_discrete_variables,
        embedding_dim,
        use_dequant_flow,
        num_blocks,
        hdims,
        base,
        actfn="swish",
        cond_embed_dim=64,
        alpha=0.01,
        arch="mlp",
        num_transformer_layers=2,
        transformer_d_model=512,
        transformer_dropout=0.0,
        block_transform="affine",
        num_mixtures=32,
        use_contextnet=True,
        flow_type="coupling",
    ):
        super().__init__()
        self.num_discrete_variables = num_discrete_variables
        self.embedding_dim = embedding_dim
        self.cond_embed_dim = cond_embed_dim
        self.use_dequant_flow = use_dequant_flow

        self.logit_centershift = layers.LogitCenterShift(alpha)
        self.log_softmax = layers.InvertibleLogSoftmax()
        self.logits_to_probs = layers.LogitsToProbabilities()
        self.reshape = Reshape(num_discrete_variables, embedding_dim)
        self.cond_logits = layers.ConditionalLogits()

        self.arch = arch
        if arch == "mlp":
            actnorm_dim = num_discrete_variables * embedding_dim
        else:
            actnorm_dim = embedding_dim

        block_fn = invblock if flow_type == "coupling" else autoreg_invblock

        if self.use_dequant_flow:
            variational_modules = []
            for i in range(num_blocks):
                variational_modules.append(layers.ActNorm1d(actnorm_dim))
                variational_modules.append(
                    block_fn(
                        i,
                        num_discrete_variables,
                        embedding_dim,
                        hdims=hdims,
                        transformer_d_model=transformer_d_model,
                        transformer_dropout=transformer_dropout,
                        num_transformer_layers=num_transformer_layers,
                        actfn=actfn,
                        cond_dim=cond_embed_dim,
                        arch=arch,
                        block_transform=block_transform,
                        num_mixtures=num_mixtures,
                    )
                )

            # inverse here because sampling uses the inverse (for actnorm init).
            self.flow = layers.Inverse(layers.SequentialFlow(variational_modules))
            self.embed = (
                nn.Embedding(embedding_dim + 1, cond_embed_dim)
                if self.cond_embed_dim is not None
                else None
            )
        else:
            self.flow = None
            self.embed = None

        self.base = base

    def _get_embeddings(self, x):
        if self.embed is None:
            return None
        return (
            self.embed(x)
            .contiguous()
            .reshape(*x.shape[:-1], self.num_discrete_variables * self.cond_embed_dim)
            .contiguous()
        )

    def forward(self, p, x, logits=True):
        z = p
        logdiff = torch.zeros(p.shape[0], 1, device=p.device)
        if not logits:
            z, logdiff = self.logits_to_probs.inverse(z, logdiff)
        z, logdiff = self.logits_centershift.inverse(z, logp=logdiff)
        z, logdiff = self.log_softmax.inverse(z, logp=logdiff)
        z, logdiff = self.cond_logits(z, x, logp=logdiff)

        if self.arch == "mlp":
            z, logdiff = self.reshape(z, logp=logdiff)
            if self.use_dequant_flow:
                cond = self._get_embeddings(x)
                z, logdiff = self.flow(z, cond=cond, logp=logdiff)
        else:
            if self.use_dequant_flow:
                cond = self._get_embeddings(x).reshape(
                    -1, self.num_discrete_variables, self.cond_embed_dim
                )
                z, logdiff = self.flow(z, cond=cond, logp=logdiff)
            z, logdiff = self.reshape(z, logp=logdiff)

        logpz = self.base(z).reshape(z.shape[0], -1).sum(1, keepdim=True)
        return logpz - logdiff

    def sample(self, x, logits=True, return_logpdf=False):
        device = x.device
        z, logpz = self.base.sample(x.shape[0], device=device)

        if self.arch == "mlp":
            if self.use_dequant_flow:
                cond = self._get_embeddings(x)
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)
            z, logpz = self.reshape.inverse(z, logp=logpz)

        else:
            z, logpz = self.reshape.inverse(z, logp=logpz)
            if self.use_dequant_flow:
                cond = self._get_embeddings(x).reshape(
                    -1, self.num_discrete_variables, self.cond_embed_dim
                )
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)

        z, logpz = self.cond_logits(z, x, logp=logpz)
        z, logpz = self.log_softmax(z, logp=logpz)
        z, logpz = self.logit_centershift(z, logp=logpz)

        if not logits:
            z, logpz = self.logits_to_probs(z, logp=logpz)
        if return_logpdf:
            return z, logpz
        else:
            return z


class VoronoiFlowDistribution(nn.Module):
    """
    A normalizing flow defined on a region with box constraints corresponding to the VoronoiTransform.
    """

    def __init__(
        self,
        voronoi_transform,
        num_discrete_variables,
        embedding_dim,
        num_blocks,
        hdims,
        base,
        actfn="swish",
        use_logit_transform=True,
        arch="mlp",
        num_transformer_layers=2,
        transformer_d_model=512,
        transformer_dropout=0.0,
        block_transform="affine",
        num_mixtures=32,
        flow_type="coupling",
    ):
        super().__init__()

        self.num_discrete_variables = num_discrete_variables
        self.embedding_dim = embedding_dim
        self.voronoi_transform = voronoi_transform
        self.use_logit_transform = use_logit_transform
        self.arch = arch
        if use_logit_transform:
            self.logit_transform = layers.LogitTransform(alpha=0.01)
        else:
            self.logit_transform = None

        self.arch = arch

        flat_dim = num_discrete_variables * embedding_dim

        block_fn = invblock if flow_type == "coupling" else autoreg_invblock

        modules = []
        for i in range(num_blocks):
            if arch != "mlp":
                modules.append(Reshape(num_discrete_variables, embedding_dim))
            modules.append(layers.ActNorm1d(flat_dim))
            if arch != "mlp":
                modules.append(
                    layers.Inverse(Reshape(num_discrete_variables, embedding_dim))
                )
            modules.append(
                block_fn(
                    i,
                    num_discrete_variables,
                    embedding_dim,
                    hdims=hdims,
                    transformer_d_model=transformer_d_model,
                    transformer_dropout=transformer_dropout,
                    num_transformer_layers=num_transformer_layers,
                    actfn=actfn,
                    arch=arch,
                    block_transform=block_transform,
                    num_mixtures=num_mixtures,
                )
            )

        self.flow = layers.SequentialFlow(modules)
        self.base = base

    def extra_repr(self):
        return (
            "num_discrete_variables={num_discrete_variables}"
            ", embedding_dim={embedding_dim}"
            ", use_logit_transform={use_logit_transform}"
        ).format(**self.__dict__)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """Assume x is shape (B, N, D) and constrained within the box constraints of the voronoi transform."""
        B, N, D = x.shape

        logdiff = x.new_zeros(B, 1)
        if self.logit_transform is not None:
            # Transform from box constraints to [0, 1]
            max_val = self.voronoi_transform.box_constraints_max.reshape(1, N, D)
            min_val = self.voronoi_transform.box_constraints_min.reshape(1, N, D)
            x = (x - min_val) / (max_val - min_val)
            logdet = (
                -torch.log(max_val - min_val)
                .reshape(1, N * D)
                .expand(B, N * D)
                .sum(1, keepdim=True)
            )
            logdiff = logdiff - logdet

            # Apply logit transform to transform into R^d
            x, logdiff = self.logit_transform(x, logp=logdiff)

        if self.arch == "mlp":
            x = x.reshape(B, N * D)
            x, logdiff = self.flow(x, logp=logdiff)
        else:
            x, logdiff = self.flow(x, logp=logdiff)
            x = x.reshape(B, N * D)

        logpz = self.base(x).reshape(x.shape[0], -1).sum(1, keepdim=True)
        return logpz - logdiff

    def sample(self, num_samples, return_logpdf=False):
        N, D = self.num_discrete_variables, self.embedding_dim
        z, logpz = self.base.sample(num_samples, device=self.device,)

        if self.arch == "mlp":
            z, logpz = self.flow.inverse(z, logp=logpz)
            z = z.reshape(num_samples, N, D)
        else:
            z = z.reshape(num_samples, N, D)
            z, logpz = self.flow.inverse(z, logp=logpz)

        # Apply inverse of logit transform to transform into [0, 1]
        if self.logit_transform is not None:
            z, logpz = self.logit_transform.inverse(z, logp=logpz)

            # Transform from [0, 1] to box constraints
            max_val = self.voronoi_transform.box_constraints_max.reshape(1, N, D)
            min_val = self.voronoi_transform.box_constraints_min.reshape(1, N, D)
            z = z * (max_val - min_val) + min_val
            logdet = (
                torch.log(max_val - min_val)
                .reshape(1, N * D)
                .expand(num_samples, N * D)
                .sum(1, keepdim=True)
            )
            logpz = logpz - logdet

        if return_logpdf:
            return z, logpz
        else:
            return z


class ConditionalVoronoiFlowDistribution(nn.Module):
    def __init__(
        self,
        voronoi_transform,
        num_discrete_variables,
        num_classes,
        embedding_dim,
        use_dequant_flow,
        num_blocks,
        hdims,
        base,
        actfn="swish",
        cond_embed_dim=64,
        share_embeddings=True,
        arch="mlp",
        num_transformer_layers=2,
        transformer_d_model=512,
        transformer_dropout=0.0,
        block_transform="affine",
        num_mixtures=32,
        use_contextnet=False,
        flow_type="coupling",
    ):
        assert isinstance(voronoi_transform, voronoi.VoronoiTransform)
        super().__init__()

        self.num_discrete_variables = num_discrete_variables
        self.num_classes = num_classes
        self.use_dequant_flow = use_dequant_flow
        self.cond_embed_dim = cond_embed_dim
        self.share_embeddings = share_embeddings

        self.embedding_dim = embedding_dim
        self.voronoi_transform = voronoi_transform

        self.arch = arch
        self.d_model = transformer_d_model

        block_fn = invblock if flow_type == "coupling" else autoreg_invblock

        if self.use_dequant_flow:
            variational_modules = []
            for i in range(num_blocks):
                if arch != "mlp":
                    variational_modules.append(
                        Reshape(num_discrete_variables, embedding_dim)
                    )
                variational_modules.append(
                    layers.ActNorm1d(num_discrete_variables * embedding_dim)
                )
                variational_modules.append(
                    layers.ConditionalAffine1d(
                        num_discrete_variables * embedding_dim,
                        nn.Sequential(
                            nn.Linear(
                                num_discrete_variables * self.cond_embed_dim, 128
                            ),
                            nn.GELU(),
                            nn.Linear(128, 128),
                            nn.GELU(),
                            nn.Linear(128, num_discrete_variables * embedding_dim * 2),
                        ),
                    )
                )
                if arch != "mlp":
                    variational_modules.append(
                        layers.Inverse(Reshape(num_discrete_variables, embedding_dim))
                    )
                variational_modules.append(
                    block_fn(
                        i,
                        num_discrete_variables,
                        embedding_dim,
                        hdims=hdims,
                        transformer_d_model=self.d_model,
                        transformer_dropout=transformer_dropout,
                        num_transformer_layers=num_transformer_layers,
                        actfn=actfn,
                        cond_dim=cond_embed_dim,
                        arch=arch,
                        block_transform=block_transform,
                        num_mixtures=num_mixtures,
                    )
                )

            # inverse here because sampling uses the inverse (for actnorm init).
            self.flow = layers.Inverse(layers.SequentialFlow(variational_modules))
        else:
            self.flow = None

        _dim = self.d_model if use_contextnet else cond_embed_dim

        if self.share_embeddings:
            self.cond_embed = nn.Embedding(self.num_classes, _dim)
        else:
            self.cond_embed = nn.ModuleList(
                [
                    nn.Embedding(self.num_classes, _dim)
                    for _ in range(num_discrete_variables)
                ]
            )

        if use_contextnet:
            if arch == "lstm":
                self.contextnet = LSTM(
                    num_layers=2,
                    d_in=None,
                    d_model=self.d_model,
                    d_out=cond_embed_dim,
                    skip_linear1=True,
                )
            else:
                self.contextnet = Transformer(
                    num_layers=2,
                    seqlen=num_discrete_variables,
                    d_in=None,
                    d_model=self.d_model,
                    d_out=cond_embed_dim,
                    dropout=0.0,
                    actfn=actfn,
                    skip_linear1=True,
                )
        else:
            self.contextnet = None

        self.base = base

    def _get_embeddings(self, x):
        if self.cond_embed is None:
            return None

        _dim = self.d_model if self.contextnet is not None else self.cond_embed_dim

        if self.share_embeddings:
            embeddings = self.cond_embed(x)
        else:
            embeddings = torch.cat(
                [
                    self.cond_embed[i](x[..., i]).reshape(-1, 1, _dim)
                    for i in range(self.num_discrete_variables)
                ],
                dim=1,
            )

        # Pass through context network.
        if self.contextnet is not None:
            embeddings = self.contextnet(embeddings)

        return embeddings

    def sample(self, x, return_logpdf=False):
        device = x.device
        N, K, D = self.num_discrete_variables, self.num_classes, self.embedding_dim
        cond = self._get_embeddings(x).reshape(-1, N * self.cond_embed_dim)
        z, logpz = self.base.sample(x.shape[0], context=cond, device=device)

        if self.arch == "mlp":
            if self.use_dequant_flow:
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)
            z = z.reshape(-1, N, D)
        else:
            z = z.reshape(-1, N, D)
            if self.use_dequant_flow:
                cond = cond.reshape(-1, N, self.cond_embed_dim)
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)

        mask = F.one_hot(x, self.num_classes).bool()

        # Center the flow at the Voronoi cell.
        points = self.voronoi_transform.anchor_pts.reshape(1, N, K, D)
        x_k = torch.masked_select(points, mask.reshape(-1, N, K, 1)).reshape(-1, N, D)
        z = z + x_k
        # Transform into the target Voronoi cell.
        z, logdet = self.voronoi_transform.map_onto_cell(z, mask=mask)
        logpz = logpz - logdet.sum(1, keepdim=True)

        if return_logpdf:
            return z, logpz
        else:
            return z


class ConditionalArgmaxFlowDistribution(nn.Module):
    def __init__(
        self,
        num_discrete_variables,
        num_classes,
        use_dequant_flow,
        num_blocks,
        hdims,
        base,
        actfn="swish",
        cond_embed_dim=64,
        share_embeddings=True,
        arch="mlp",
        num_transformer_layers=2,
        transformer_d_model=512,
        transformer_dropout=0.0,
        block_transform="affine",
        num_mixtures=32,
        use_contextnet=False,
        flow_type="coupling",
    ):
        super().__init__()

        self.num_discrete_variables = num_discrete_variables
        self.num_classes = num_classes
        self.use_dequant_flow = use_dequant_flow
        self.cond_embed_dim = cond_embed_dim
        self.share_embeddings = share_embeddings

        self.embedding_dim = int(np.ceil(np.log2(num_classes)))
        self.softplus = layers.Softplus()

        self.arch = arch
        self.d_model = transformer_d_model

        block_fn = invblock if flow_type == "coupling" else autoreg_invblock

        if self.use_dequant_flow:
            variational_modules = []
            for i in range(num_blocks):
                if arch != "mlp":
                    variational_modules.append(
                        Reshape(num_discrete_variables, self.embedding_dim)
                    )
                variational_modules.append(
                    layers.ActNorm1d(num_discrete_variables * self.embedding_dim)
                )
                if arch != "mlp":
                    variational_modules.append(
                        layers.Inverse(
                            Reshape(num_discrete_variables, self.embedding_dim)
                        )
                    )
                variational_modules.append(
                    block_fn(
                        i,
                        num_discrete_variables,
                        self.embedding_dim,
                        hdims=hdims,
                        transformer_d_model=self.d_model,
                        transformer_dropout=transformer_dropout,
                        num_transformer_layers=num_transformer_layers,
                        actfn=actfn,
                        cond_dim=cond_embed_dim,
                        arch=arch,
                        block_transform=block_transform,
                        num_mixtures=num_mixtures,
                    )
                )

            # inverse here because sampling uses the inverse (for actnorm init).
            self.flow = layers.Inverse(layers.SequentialFlow(variational_modules))
        else:
            self.flow = None

        _dim = self.d_model if use_contextnet else cond_embed_dim

        if self.share_embeddings:
            self.cond_embed = nn.Embedding(self.num_classes, _dim)
        else:
            self.cond_embed = nn.ModuleList(
                [
                    nn.Embedding(self.num_classes, _dim)
                    for _ in range(num_discrete_variables)
                ]
            )

        if use_contextnet:
            if arch == "lstm":
                self.contextnet = LSTM(
                    num_layers=2,
                    d_in=None,
                    d_model=self.d_model,
                    d_out=cond_embed_dim,
                    skip_linear1=True,
                )
            else:
                self.contextnet = Transformer(
                    num_layers=2,
                    seqlen=num_discrete_variables,
                    d_in=None,
                    d_model=self.d_model,
                    d_out=cond_embed_dim,
                    dropout=0.0,
                    actfn=actfn,
                    skip_linear1=True,
                )
        else:
            self.contextnet = None

        self.base = base

    def _get_embeddings(self, x):
        if self.cond_embed is None:
            return None

        _dim = self.d_model if self.contextnet is not None else self.cond_embed_dim

        if self.share_embeddings:
            embeddings = self.cond_embed(x)
        else:
            embeddings = torch.cat(
                [
                    self.cond_embed[i](x[..., i]).reshape(-1, 1, _dim)
                    for i in range(self.num_discrete_variables)
                ],
                dim=1,
            )

        # Pass through context network.
        if self.contextnet is not None:
            embeddings = self.contextnet(embeddings)

        return embeddings

    def sample(self, x, return_logpdf=False):
        device = x.device
        N, K, D = self.num_discrete_variables, self.num_classes, self.embedding_dim
        cond = self._get_embeddings(x).reshape(-1, N * self.cond_embed_dim)
        z, logpz = self.base.sample(x.shape[0], context=cond, device=device)

        if self.arch == "mlp":
            if self.use_dequant_flow:
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)
            z = z.reshape(-1, N, D)
        else:
            z = z.reshape(-1, N, D)
            if self.use_dequant_flow:
                cond = cond.reshape(-1, N, self.cond_embed_dim)
                z, logpz = self.flow.inverse(z, cond=cond, logp=logpz)

        binary = argmax_utils.integer_to_base(x, base=2, dims=self.embedding_dim)
        sign = binary * 2 - 1

        z, logpz = self.softplus(z, logp=logpz)
        z = z * sign

        if return_logpdf:
            return z, logpz
        else:
            return z


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return x * torch.sigmoid_(x * F.softplus(self.beta))


actfns = {
    "swish": Swish,
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def MLP(dims, actfn):
    modules = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(actfns[actfn]())
    return nn.Sequential(*modules[:-1])


def build_MLP(d_in, hdims, d_out, actfn="swish"):
    dims = [d_in] + list(hdims) + [d_out]
    return MLP(dims, actfn)


def invblock(
    i,
    num_discrete_variables,
    embedding_dim,
    *,
    hdims,
    num_transformer_layers=2,
    transformer_d_model=512,
    transformer_dropout=0.0,
    actfn="swish",
    cond_dim=0,
    arch="mlp",
    block_transform="affine",
    num_mixtures=32,
):
    mask_type = {0: "skip0", 1: "skip1", 2: "channel0", 3: "channel1",}[i % 4]
    if block_transform == "affine":
        block_fn = layers.MaskedCouplingBlock
        out_factor = 2
    else:
        block_fn = partial(
            layers.MixLogCDFCouplingBlock,
            num_mixtures=num_mixtures,
            dim_size=embedding_dim
            if arch != "mlp"
            else embedding_dim * num_discrete_variables,
        )
        out_factor = 2 + 3 * num_mixtures

    if arch == "mlp":
        return block_fn(
            build_MLP(
                num_discrete_variables * (embedding_dim + cond_dim),
                hdims,
                out_factor * num_discrete_variables * embedding_dim,
                actfn,
            ),
            mask_dim=1,
            mask_type=mask_type,
        )
    elif arch == "transformer":
        return block_fn(
            Transformer(
                num_transformer_layers,
                seqlen=num_discrete_variables,
                d_in=embedding_dim,
                d_out=out_factor * embedding_dim,
                d_model=transformer_d_model,
                dropout=transformer_dropout,
                actfn=actfn,
                cond_dim=cond_dim,
            ),
            mask_dim=2,
            mask_type=mask_type,
        )
    elif arch == "lstm":
        return block_fn(
            LSTM(
                num_transformer_layers,
                d_in=embedding_dim,
                d_out=out_factor * embedding_dim,
                d_model=transformer_d_model,
                cond_dim=cond_dim,
            ),
            mask_dim=2,
            mask_type=mask_type,
        )
    else:
        raise ValueError(f"Unknown architecture {arch}")


def autoreg_invblock(
    i,
    num_discrete_variables,
    embedding_dim,
    *,
    hdims,
    actfn="swish",
    cond_dim=0,
    arch="mlp",
    block_transform="affine",
    **kwargs,
):

    assert arch == "mlp"
    assert block_transform == "affine"

    return layers.AffineAutoregressive(
        MADE(
            nin=num_discrete_variables * embedding_dim,
            hidden_sizes=list(hdims),
            nout=2 * num_discrete_variables * embedding_dim,
            ncond=num_discrete_variables * cond_dim,
            actfn=actfns[actfn],
        ),
    )


class LSTM(nn.Module):
    def __init__(
        self, num_layers, d_in, d_out, d_model=512, cond_dim=0, skip_linear1=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.module = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        if skip_linear1:
            self.linear1 = None
        else:
            self.linear1 = nn.Linear(d_in + cond_dim, d_model)
        self.linear2 = nn.Linear(2 * d_model, d_out)

    def forward(self, x):
        if len(x.shape) == 2:
            reshape = True
            B = x.shape[0]
            x = x.reshape(B, self.seqlen, -1)
        else:
            reshape = False

        B, L, _ = x.shape

        if self.linear1 is not None:
            x = self.linear1(x)
        x = self.module(x)[0]
        x = self.linear2(x)

        if reshape:
            x = x.reshape(B, -1)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        seqlen,
        d_in,
        d_out,
        d_model=512,
        nhead=4,
        dim_feedforward=512,
        dropout=0.0,
        actfn="relu",
        cond_dim=0,
        skip_linear1=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.seqlen = seqlen
        self.embed_positions = nn.Embedding(seqlen, d_model)
        self.layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=actfns[actfn](),
                )
                for _ in range(num_layers)
            ]
        )
        if skip_linear1:
            self.linear1 = None
        else:
            self.linear1 = nn.Linear(d_in + cond_dim, d_model)
        self.linear2 = nn.Linear(d_model, d_out)

    def forward(self, x):

        if len(x.shape) == 2:
            reshape = True
            B = x.shape[0]
            x = x.reshape(B, self.seqlen, -1)
        else:
            reshape = False

        B, L, _ = x.shape

        # Convert from (B, L, D) to (L, B, D).
        x = x.transpose(0, 1)

        if self.linear1 is not None:
            x = self.linear1(x)

        positions = self.embed_positions(torch.arange(L).to(x.device).reshape(-1, 1))
        x = x + positions
        for mod in self.layers:
            x = mod(x)
        x = self.linear2(x)

        # Convert back.
        x = x.transpose(0, 1)

        if reshape:
            x = x.reshape(B, -1)
        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = layers.ActNorm1d(d_model)
        self.norm2 = layers.ActNorm1d(d_model)

        # self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)
        # self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        if isinstance(activation, str):
            self.activation = actfns[activation]()
        else:
            self.activation = activation

    def forward(self, src):
        x = src
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x):
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def get_sinusoidal_embedding(num_embeddings, embedding_dim):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
        0
    )
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    return emb

