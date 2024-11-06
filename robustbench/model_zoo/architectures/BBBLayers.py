import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# from metrics import calculate_kl as KL_DIV

from torch import nn

RHO = -4

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False, priors=None):
        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.posterior_mu_initial = (0, 0.1)
        self.posterior_rho_initial = (RHO, 0.0)

        self.weight_mu = Parameter(torch.empty((out_channels, in_channels//self.groups, *self.kernel_size), device=self.device))
        self.weight_rho = Parameter(torch.empty((out_channels, in_channels//self.groups, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training and sample:
            W_eps = torch.empty(self.weight_mu.size()).normal_(0, 1).to(self.device)
            self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + W_eps * self.weight_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_weight_mu, self.prior_weight_rho, self.weight_mu, self.weight_rho)
        if self.use_bias:
            kl += KL_DIV(self.prior_bias_mu, self.prior_bias_rho, self.bias_mu, self.bias_rho)
        return kl
    
    def compute_kl_with_prior(self, prior_module=None):

        if prior_module != None:
            prior_weight_mu = prior_module.weight_mu
            prior_weight_rho = prior_module.weight_rho
            if self.use_bias:
                prior_bias_mu = prior_module.bias_mu
                prior_bias_rho = prior_module.bias_rho
        else:
            priors = { 'prior_mu': 0, 'prior_rho': 0.1 }
            prior_weight_mu = priors['prior_mu']
            prior_weight_rho = priors['prior_rho']
            if self.use_bias:
                prior_bias_mu = priors['prior_mu']
                prior_bias_rho = priors['prior_rho']

        kl = KL_DIV(prior_weight_mu, prior_weight_rho, self.weight_mu, self.weight_rho)
        if self.use_bias:
            kl += KL_DIV(prior_bias_mu, prior_bias_rho, self.bias_mu, self.bias_rho)

        return kl
    
        
    
class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.posterior_mu_initial = (0, 0.1)
        self.posterior_rho_initial = (RHO, 0.0)

        self.weight_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.weight_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training and sample:
            W_eps = torch.empty(self.weight_mu.size()).normal_(0, 1).to(self.device)
            self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + W_eps * self.weight_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                self.bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.weight_mu, self.weight_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
    
    def compute_kl_with_prior(self, prior_module=None):

        if prior_module != None:
            prior_weight_mu = prior_module.weight_mu
            prior_weight_rho = prior_module.weight_rho
            if self.use_bias:
                prior_bias_mu = prior_module.bias_mu
                prior_bias_rho = prior_module.bias_rho
        else:
            priors = { 'prior_mu': 0, 'prior_rho': 0.1 }
            prior_weight_mu = priors['prior_mu']
            prior_weight_rho = priors['prior_rho']
            if self.use_bias:
                prior_bias_mu = priors['prior_mu']
                prior_bias_rho = priors['prior_rho']

        kl = KL_DIV(prior_weight_mu, prior_weight_rho, self.weight_mu, self.weight_rho)
        if self.use_bias:
            kl += KL_DIV(prior_bias_mu, prior_bias_rho, self.bias_mu, self.bias_rho)

        return kl
    
def KL_DIV(mu_q, rho_q, mu_p, rho_p):
    sig_q = torch.log1p(torch.exp(rho_q))
    sig_p = torch.log1p(torch.exp(rho_p))
    kl = 0.5 * (2 * torch.log(sig_q / sig_p) - 1 + (sig_p / sig_q).pow(2) + ((mu_p - mu_q) / sig_q).pow(2)).sum()
    return kl

