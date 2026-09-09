"""STS-GCN human-motion forecaster.

Adapted from the official STS-GCN implementation (Sofianos et al., ICCV 2021,
https://github.com/FraLuca/STSGCN, MIT) — the same backbone used by ManiCast
(Kedia et al., CoRL 2023). Cleaned up, no functional changes: a stack of
space-time separable graph-conv layers followed by a temporal-extrapolation
CNN that maps T_in history frames to T_out future frames.

Shapes: input (N, C=3, T_in, V joints) -> output (N, T_out, C=3, V).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """Separable space-time graph convolution with learnable adjacencies."""

    def __init__(self, time_dim: int, joints_dim: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(time_dim, joints_dim, joints_dim))
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)
        self.T = nn.Parameter(torch.empty(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        x = torch.einsum("nctv,tvw->nctw", (x, self.A))
        return x.contiguous()


class STGCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 time_dim, joints_dim, dropout):
        super().__init__()
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.gcn = ConvTemporalGraphical(time_dim, joints_dim)
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      (kernel_size[0], kernel_size[1]),
                      (stride, stride), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.prelu(x + res)


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class STSGCN(nn.Module):
    def __init__(self, input_channels=3, input_time_frame=8,
                 output_time_frame=20, st_gcnn_dropout=0.1,
                 joints_to_consider=9, n_txcnn_layers=4,
                 txc_kernel_size=(3, 3), txc_dropout=0.0):
        super().__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.st_gcnns = nn.ModuleList([
            STGCNNLayer(input_channels, 64, [1, 1], 1, input_time_frame,
                        joints_to_consider, st_gcnn_dropout),
            STGCNNLayer(64, 32, [1, 1], 1, input_time_frame,
                        joints_to_consider, st_gcnn_dropout),
            STGCNNLayer(32, 64, [1, 1], 1, input_time_frame,
                        joints_to_consider, st_gcnn_dropout),
            STGCNNLayer(64, input_channels, [1, 1], 1, input_time_frame,
                        joints_to_consider, st_gcnn_dropout),
        ])
        self.txcnns = nn.ModuleList(
            [CNNLayer(input_time_frame, output_time_frame, txc_kernel_size,
                      txc_dropout)]
            + [CNNLayer(output_time_frame, output_time_frame, txc_kernel_size,
                        txc_dropout) for _ in range(n_txcnn_layers - 1)]
        )
        self.prelus = nn.ModuleList(
            [nn.PReLU() for _ in range(n_txcnn_layers)])

    def forward(self, x):
        # x: (N, 3, T_in, V)
        for gcn in self.st_gcnns:
            x = gcn(x)
        x = x.permute(0, 2, 1, 3)  # (N, T_in, 3, V)
        x = self.prelus[0](self.txcnns[0](x))
        for i in range(1, len(self.txcnns)):
            x = self.prelus[i](self.txcnns[i](x)) + x
        return x  # (N, T_out, 3, V)
