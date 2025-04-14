#!/usr/bin/env python
"""
Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and Technology
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

# TCN 

class accelEncTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(accelEncTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out.permute(0, 2, 1)
        
        out = self.linear(out[:, -1:, :])
        return out

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linear_vel = nn.Linear(num_channels[-1], output_size)
        self.linear_std = nn.Linear(num_channels[-1], output_size)
        
        self.vel_decoder = nn.Linear(3, num_channels[-1])
        
    def forward(self, x, vel_init= None):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out.permute(0, 2, 1)
        
        if vel_init is not None:
            h0_ = self.vel_decoder(vel_init).permute(1,0,2)
        
        vel = self.linear_vel(out[:, :, :])
        std = self.linear_std(out[:, :, :])
        return vel, std

class RNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, real_time=False, skip_connection=False):
        super(RNN, self).__init__()
        self.layers = len(num_channels)
        self.rnn = nn.GRU(input_size, num_channels[-1], self.layers, dropout=dropout, batch_first=True).cuda()
        
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation_skip = nn.Tanh()
        
        self.input_decoder = nn.Linear(input_size, num_channels[-1])
        self.vel_decoder0 = nn.Linear(3, num_channels[-1]) 
        self.vel_decoder1 = nn.Linear(3, num_channels[-1]) 
        self.vel_decoder2 = nn.Linear(3, num_channels[-1]) 
        
        self.linear_vel = nn.Linear(num_channels[-1], output_size)
        self.linear_std = nn.Linear(num_channels[-1], output_size)
        
        self.rnn_vel_initialized = False
        self.real_time = real_time        
        self.skip_connection = skip_connection
        
        if self.skip_connection:
            self.skip_attention = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x, h0=None, vel_init= None):
        # if vel_init is not None:
        if vel_init is not None and h0 is None:
            # print("Velocity Init ####################################################")
            h0_0 = self.vel_decoder0(vel_init).permute(1,0,2)                                                                         # ToDo : replace with seperate decoder for each layer
            h0_1 = self.vel_decoder1(vel_init).permute(1,0,2)                                                                         # ToDo : replace with seperate decoder for each layer
            h0_2 = self.vel_decoder2(vel_init).permute(1,0,2)                                                                         # ToDo : replace with seperate decoder for each layer
            h0 = torch.cat([h0_0, h0_1, h0_2], axis = 0)

            self.rnn_vel_initialized = True
            
        out, h = self.rnn(x, h0)
        out = self.activation1(out)
        
        if self.skip_connection:
            skip_atten_weights = F.softmax(self.skip_attention(out), dim = 1)
            out = out + skip_atten_weights*self.input_decoder(x)         # Skip connection
        
        vel = self.linear_vel(out)
        std = self.linear_std(out)
        
        if self.real_time:
            return vel[:, -1:], std[:, -1:], h
        else:
            return vel, std

class RNN2(nn.Module):
    def __init__(self, input_imu_size, input_imu_len, input_act_size, output_size, num_channels, kernel_size, dropout):
        super(RNN2, self).__init__()
        self.layers = 2
        output_imu_size = 16
        self.input_imu_len = input_imu_len
        self.rnn = nn.LSTM(input_act_size+output_imu_size, num_channels[-1], self.layers, dropout=dropout, batch_first=True).cuda()
        self.rnn_imu_encoder = nn.GRU(input_imu_size, output_imu_size, 2, dropout=dropout, batch_first=True).cuda()
        
        self.activation = nn.LeakyReLU(0.1)
        self.vel_biases_decoder = nn.Linear(3+6, num_channels[-1])
        
        self.linear_vel = nn.Linear(num_channels[-1], output_size)
        self.linear_std = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, act, imu, h0=None, vel_init= None):
        if vel_init is not None:
            h0_ = self.vel_biases_decoder(vel_init).permute(1,0,2)
            h0 = torch.cat([h0_]*self.layers, axis = 0)
        
        imu_latent_list = []
        imu = imu.view(act.shape[0], -1, 10, 6)
        for i in range(act.shape[1]):
            out_, _ = self.rnn_imu_encoder(imu[:,i])
            imu_latent_list.append(out_[:,-1:])
        
        imu_latent = torch.cat(imu_latent_list, dim = 1)
        
        x = torch.cat([imu_latent, act], dim = -1)
        out, h_c = self.rnn(x, h0)
        # h = h.permute(1,0,2)
        # out = self.activation(out)
        vel = self.linear_vel(out[:, -1:, :])
        std = self.linear_std(self.activation(out[:, -1:, :]))
        
        return vel, std, torch.sum(self.linear_vel(out[:, :, :]), dim=1)

