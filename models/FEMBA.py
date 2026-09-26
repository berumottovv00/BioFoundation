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

import torch
import torch.nn as nn
from typing import Optional, Tuple
from mamba_ssm import Mamba


class MambaWrapper(nn.Module):
    """
    Thin wrapper around Mamba to support bi-directionality.

    Args:
        d_model (int): Dimension of the model.
        bidirectional (bool): Whether to use bidirectional processing.
        bidirectional_strategy (str, optional): Strategy to combine forward and backward passes ("add", "ew_multiply").
        **mamba_kwargs: Additional arguments passed to Mamba.

    中文说明：
    对 mamba_ssm.Mamba 的简单封装，让它支持"双向"处理——正向跑一遍序列，
    反向（flip 之后）再跑一遍，再按 bidirectional_strategy 把两路结果合并
    （"add" 相加 或 "ew_multiply" 逐元素相乘）。单向的 Mamba 本身只能看到
    "过去"的信息，双向包一层之后每个位置能同时看到前后文。
    """
    def __init__(self, d_model: int, bidirectional: bool = True, bidirectional_strategy: Optional[str] = "add", **mamba_kwargs):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"{bidirectional_strategy} strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        # 正向 Mamba：按序列原本的顺序处理
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            # 反向 Mamba：权重独立的另一份 Mamba，处理时序列会先 flip 再喂进来
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            # 序列维度（dim=1）先 flip 再过 mamba_rev，处理完再 flip 回来对齐原始顺序
            out_rev = self.mamba_rev(hidden_states.flip(dims=(1,)), inference_params=inference_params).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
        return out


class PatchEmbed(nn.Module):
    """
    Converts input signal into patch embeddings using a convolutional layer.

    Args:
        inp_size (Tuple[int, int]): Input size (channels, sequence length).
        patch_size (Tuple[int, int]): Size of each patch.
        stride (Tuple[int, int]): Stride for patch extraction.
        in_chans (int): Number of input channels.
        embed_dim (int): Dimension of the embedding.
        kernel_1 (int): Kernel size (used to define padding).
        norm_layer (nn.Module, optional): Normalization layer.

    中文说明：
    把原始信号 (B, C, T) 当成"单通道图像"，用 Conv2d 按 patch_size/stride 切块
    并投影到 embed_dim 维，再拉平成一个序列，供后面的 Mamba encoder 处理。
    grid_size = (grid_h, grid_w) 是切完之后的网格尺寸：grid_h 对应通道方向切了
    几块，grid_w 对应时间方向切了几块（也就是最终序列长度）。
    """
    def __init__(self, inp_size, patch_size, stride, in_chans, embed_dim, kernel_1: int = 64, norm_layer=None):
        super().__init__()
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.kernel_1 = kernel_1 - 1
        # grid_size[0]: 通道方向切出多少块；grid_size[1]: 时间方向切出多少块（= 序列长度）
        self.grid_size = ((inp_size[0] - patch_size[0]) // stride[0] + 1,
                          (inp_size[1] - patch_size[1]) // stride[1] + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, channels, length) —— 加一个"通道"维度当成单通道图像
        x = self.proj(x)  # (batch, embed_dim, grid_h, grid_w) —— 卷积切块+投影
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # (batch, embed_dim * grid_h, grid_w) —— 把 embed_dim 和 grid_h 拼成 d_model
        x = x.permute(0, 2, 1)  # (batch, grid_w, embed_dim * grid_h) —— grid_w 变成序列维，d_model 变成特征维
        x = self.norm(x)
        return x


class MambaClassifier(nn.Module):
    """
    Classifier head using Mamba block for temporal processing.

    Args:
        embed_dim (int): Embedding dimension.
        grid_size (Tuple[int, int]): Grid size from patch embedding.
        num_classes (int): Number of output classes.
        num_channels (int): Number of input channels.
        classification_type (str): Classification strategy:
            - 'bc'  = Binary Classification (e.g. TUAB, TUAR)
            - 'ml'  = Multi-Label Classification (e.g. TUSL)
            - 'mc' = Multi-Label  Classification  for TUAR 
            - 'mcc' = Multi-Class Classification (e.g. TUAR )
            - 'mmc' = Multi-Class Multi-Output Classification (e.g. TUAR)

    中文说明：
    微调阶段接在 encoder 输出后面的分类头。结构是：线性层升/降维 -> GELU ->
    再过一个独立的 Mamba block 做进一步的时序建模 -> 沿时间维做全局平均池化
    压成定长向量 -> 线性层输出 logits。fc3 的输出维度按 classification_type
    不同而不同（比如 "mc" 是逐通道二分类，输出维度是 num_channels）。
    """
    def __init__(self, embed_dim, grid_size, num_classes, num_channels, classification_type: str):
        super(MambaClassifier, self).__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.classification_type = classification_type

        hidden_size1 = 256
        input_size = embed_dim * grid_size[0]

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.GELU()
        # 注意：这是分类头自己独立的一个 Mamba block，跟 encoder 里的双向 Mamba 是两码事
        self.mamba_1 = Mamba(d_model=hidden_size1, expand=2)

        # 不同分类模式对应不同的输出维度
        if classification_type in ("bc", "mcc", "ml"):
            self.fc3 = nn.Linear(hidden_size1, num_classes)
        elif classification_type == "mc":
            self.fc3 = nn.Linear(hidden_size1, num_channels)
        elif classification_type == "mmc":
            self.fc3 = nn.Linear(hidden_size1, num_channels * num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.mamba_1(x)
        x = x.permute(0, 2, 1).contiguous()  # (batch, features, time)
        x = x.mean(dim=-1)  # Global average pooling over time —— 把整个时间序列压成一个定长向量
        x = self.fc3(x)

        if self.classification_type == "mmc":
            x = x.view(-1, self.num_channels, self.num_classes)

        return x


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


class FEMBA(nn.Module):
    """
    Foundational Encoder Model with Bidirectional Mamba (FEMBA).
    Can perform classification or masked signal reconstruction.

    Args:
        seq_length (int): Length of the input signal.
        num_channels (int): Number of input channels.
        num_classes (int): Number of output classes. Set to 0 for reconstruction.
        kernel_1 (int): First convolution kernel size.
        kernel_dec (Tuple[int, int]): Decoder kernel size.
        dropout (float): Dropout rate.
        exp (int): Expansion factor for Mamba.
        patch_size (Tuple[int, int]): Patch size for encoder.
        stride (Tuple[int, int]): Stride for encoder.
        embed_dim (int): Embedding dimension.
        num_blocks (int): Number of Mamba blocks.
        classification_type (str): Classification type (bc, ml, mc, mcc, mmc).

    中文说明：
    整体结构是 PatchEmbed -> 加位置编码 -> num_blocks 层双向 Mamba encoder ->
    二选一的输出头：num_classes==0 时接 Decoder（自监督预训练，masked
    reconstruction），num_classes>0 时接 MambaClassifier（下游分类微调）。
    encoder 部分（PatchEmbed + Mamba blocks）在预训练和微调之间是共享结构，
    只是外面接的"头"不一样——这也是为什么预训练权重能迁移到微调阶段。
    """
    def __init__(self,
                seq_length: int = 1280,
                num_channels: int = 22,
                num_classes: int = 0,
                kernel_1: int = 64,
                kernel_dec: Tuple[int, int] = (31, 31),
                exp: int = 4, 
                patch_size: Tuple[int, int] = (2, 16),
                stride: Tuple[int, int] = (2, 16),
                embed_dim: int = 79,
                num_blocks: int= 1,
                classification_type: str = "bc"):

        super(FEMBA, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1 - 1
        self.exp = exp
        self.inp_size = (num_channels, seq_length)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.classification_type = classification_type

        self.patch_embed = PatchEmbed(
            inp_size=self.inp_size,
            patch_size=self.patch_size,
            stride=self.stride,
            in_chans=1,
            embed_dim=embed_dim
        )

        grid_size = self.patch_embed.grid_size
        # 可学习位置编码，形状跟 patch_embed 输出的序列 (grid_w, d_model) 对齐
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[1], grid_size[0] * self.embed_dim))

        # d_model = grid_size[0] * embed_dim，num_blocks 层双向 Mamba 堆叠组成 encoder
        self.mamba_blocks = nn.ModuleList([
            MambaWrapper(d_model=grid_size[0] * self.embed_dim, expand=self.exp)
            for _ in range(self.num_blocks)
        ])

        # 每个 Mamba block 后面配一个 LayerNorm（跟残差连接配合使用，见 forward）
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(grid_size[0] * self.embed_dim)
            for _ in range(self.num_blocks)
        ])

        # num_classes==0 -> 预训练重建分支；num_classes>0 -> 微调分类分支
        if num_classes == 0:
            self.classifier = None
            self.decoder = Decoder(embed_dim=self.embed_dim, grid_size=grid_size, kernel_dec=kernel_dec, patch_size=self.patch_size, stride=stride)
        else:
            self.classifier = MambaClassifier(embed_dim, grid_size, num_classes, num_channels, classification_type)

    def forward(self, x, mask):
        # ===== 步骤1: 输入 + 掩码 =====
        # mask 由 MaskTask.generate_mask() 在模型外生成（见 tasks/pretrain_task.py 的"步骤0"），
        # 这里只是把被选中的 patch 区域置零。x_original 保留未掩码的原始信号，用于计算重建损失。
        x_original = x
        x_masked = x.clone()
        x_masked[mask] = 0  # Apply mask

        # ===== 步骤2: Patch Embedding =====
        # Conv2d 把 (B, C, T) 切成 (patch_H, patch_W) 的块并投影到 embed_dim，
        # 再 reshape/permute 成序列 (B, seq_len=grid_w, d_model=grid_h*embed_dim)，
        # 然后加上可学习位置编码 pos_embed。
        x = self.patch_embed(x_masked)  # (B, T, D)
        x = x + self.pos_embed  # Add positional embedding

        # ===== 步骤3: Encoder（双向 Mamba × num_blocks） =====
        # 每个 block: 残差 + MambaWrapper（正向+反向 Mamba 相加）+ LayerNorm
        for mamba_block, norm_layer in zip(self.mamba_blocks, self.norm_layers):
            res = x
            x = mamba_block(x)
            x = res + x
            x = norm_layer(x)

        # ===== 步骤4: 分支 —— Decoder（预训练重建）或 Classifier（微调分类） =====
        # num_classes == 0 时走 Decoder 分支（自监督预训练）；
        # num_classes > 0 时走 MambaClassifier 分支（下游分类微调）。
        if self.classifier is not None:
            x_classified = self.classifier(x)
            # ===== 步骤5: 返回 (分类结果, 原始信号) =====
            return x_classified, x_original
        else:
            x_reconstructed = self.decoder(x)
            # ===== 步骤5: 返回 (重建信号, 原始信号) —— 供 criterion 只在被掩码区域算损失 =====
            return x_reconstructed, x_original
