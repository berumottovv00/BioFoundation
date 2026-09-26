#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
"""
models/FEMBA.py 中 Decoder 的副本，供 QAP 模型（models/FEMBA_int8.py）使用。

单独复制一份是为了不依赖 mamba_ssm：`from models.FEMBA import Decoder` 会执行
models/FEMBA.py 顶部的 `from mamba_ssm import Mamba`，而 QAP 不需要 mamba_ssm。
Decoder 只在预训练时使用、部署时丢弃，所以两份代码不同步也不会影响量化编码器的部署。
"""

import torch.nn as nn
from typing import Tuple


class Decoder(nn.Module):
    """
    Reconstructs original signal from encoded representation.

    Args:
        embed_dim (int): Embedding dimension.
        grid_size (Tuple[int, int]): Grid size from encoder.
        kernel_dec (Tuple[int, int]): Kernel size for decoding conv.
        patch_size (Tuple[int, int]): Patch size used in encoding.
        stride (Tuple[int, int]): Stride used in encoding.

    中文说明：
    预训练阶段接在 encoder 输出后面的重建头，作用跟 PatchEmbed 正好相反——
    PatchEmbed 用 Conv2d 把原始信号"压缩"成 patch 序列，Decoder 用
    ConvTranspose2d（反卷积/转置卷积）把压缩后的特征序列"放大"还原回原始
    (channels, seq_length) 的形状，重建出的信号只在被 mask 的区域跟原始信号
    比较（算损失），逼着模型学会"猜"被挡住的信号长什么样。
    """
    def __init__(self, embed_dim: int, grid_size: Tuple[int, int], kernel_dec: Tuple[int, int], patch_size: Tuple[int, int], stride: Tuple[int, int]):
        super(Decoder, self).__init__()
        self.kernel_dec = kernel_dec
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # 先用一个"same padding"的卷积做局部特征平滑/混合，不改变形状
        self.dec_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_dec,
            stride=1,
            padding=((self.kernel_dec[0] - 1) // 2, (self.kernel_dec[1] - 1) // 2),
            bias=False
        )

        # 把 (embed_dim * grid_h) 这个拼在一起的维度，重新拆回 (embed_dim, grid_h) 两个维度
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(embed_dim, grid_size[0])
        )

        # 反卷积：用跟 PatchEmbed 里 Conv2d 完全相同的 kernel_size/stride，把网格形状"放大"还原回 (channels, seq_length)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=patch_size,
            stride=stride
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, N=grid_size[0] * embed_dim, grid_size[1])
        x = self.dec_conv(x)  # (batch, 1, grid_size[0] * grid_size[1], embed_dim)
        x = x.squeeze(1)  # (batch, grid_size[0] * grid_size[1], embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, grid_size[0] * grid_size[1])
        x = self.unflatten(x)  # (batch, embed_dim, grid_size[0], grid_size[1])
        x = self.conv_transpose(x)  # (batch, 1, H, W)
        x_reconstructed = x.squeeze(1)  # (batch, H, W)
        return x_reconstructed
