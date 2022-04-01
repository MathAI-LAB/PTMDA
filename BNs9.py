#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 07:53:09 2020

@author: user1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SBatchNorm2d(nn.Module):
    """Batch Normalization implented by vanilla Python from scratch. It is a startkit for your own idea.
    Parameters
        num_features ¨C C from an expected input of size (N, C, H, W)
        eps ¨C a value added to the denominator for numerical stability. Default: 1e-5
        momentum ¨C the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1

    Shape:
        Input: (N, C, H, W)
        Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> m = BatchNorm2d(100)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_meanS', torch.zeros(num_features))
            self.register_buffer('running_varS', torch.ones(num_features))
            self.register_buffer('running_meanT', torch.zeros(num_features))
            self.register_buffer('running_varT', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_meanS', None)
            self.register_parameter('running_varS', None)
            self.register_parameter('running_meanT', None)
            self.register_parameter('running_varT', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_meanS.zero_()
            self.running_varS.fill_(1)
            self.running_meanT.zero_()
            self.running_varT.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]# 
            input_target = input[batch_size:]#
            
            meanSo = input_source.mean([0, 2, 3])
            varSo = input_source.var([0, 2, 3], unbiased=False)

            meanT = input_target.mean([0, 2, 3])
            varT = input_target.var([0, 2, 3], unbiased=False)
            
            
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_meanS = exponential_average_factor * meanSo\
                    + (1 - exponential_average_factor) * self.running_meanS

                self.running_varS = exponential_average_factor * varSo * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_varS
                self.running_meanT = exponential_average_factor * meanT\
                    + (1 - exponential_average_factor) * self.running_meanT

                self.running_varT = exponential_average_factor * varT * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_varT
            #print("**self.bias**",self.bias.requires_grad)
            weightS=self.weight.detach()
            biasS=self.bias.detach()
            Ys = (input_source - meanSo[None, :, None, None]) / (torch.sqrt(varSo[None, :, None, None] + self.eps))
            Ys = Ys * weightS[None, :, None, None] + biasS[None, :, None, None]
            
            Yt = (input_target - meanT[None, :, None, None]) / (torch.sqrt(varT[None, :, None, None] + self.eps))

            Yt = Yt * self.weight[None, :, None, None] + self.bias[None, :, None, None]


            #print("biasT", biasT.requires_grad)
            return torch.cat((Ys, Yt), dim=0)
        else:
            mean = self.running_meanT
            var = self.running_varT

            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

            return input
