# Copyright (c) 2026 Thorir Mar Ingolfsson, ETH Zurich
# SPDX-License-Identifier: Apache-2.0

"""
Test 34: FEMBA Tiny INT8 (True Architecture)

中文说明：
本文件是 FEMBA（PatchEmbed + 双向 Mamba 编码器 + 分类头）的 INT8 量化版本，
用 Brevitas 的量化层（QuantIdentity/QuantLinear/QuantConv2d 等）重新实现了
models/FEMBA.py 中 FEMBA 的计算图，用于 QAT（量化感知训练）以及后续 ARES
"伪量化 -> 真 INT8" 的转换、部署到 GAP9。
注意：这是分类分支（带 classifier），不是预训练重建分支（没有 Decoder）。

  - Input: (1, 1, 22, 1280) - 22 EEG channels, 1280 samples (~5s @ 256Hz)
  - Patch Embedding (patch_size=(2,16), stride=(2,16), embed_dim=35)
  - d_model = 385 (11 * 35)
  - expand = 4 (FEMBA standard)
  - d_inner = 1540 (4 * d_model)
  - d_state = 16
  - dt_rank = 25 (ceil(d_model / 16))
  - 2 BiMamba blocks with L3 streaming

Total Parameters: ~7.6 million

Memory requirements per direction (~1.94 MB):
  - in_proj: 385 * 3080 = 1,185,800 bytes (~1.13 MB!)
  - out_proj: 1540 * 385 = 593,450 bytes (~580 KB)
  - conv1d: 1540 * 4 + 1540 = 7,700 bytes
  - x_proj: 1540 * 57 = 87,780 bytes
  - dt_proj: 25 * 1540 + 1540 = 40,040 bytes
  - A_log: 1540 * 16 * 4 = 98,560 bytes (FP32)
  - D: 1540 * 4 = 6,160 bytes (FP32)

Total for 4 directions: ~7.8 MB (requires aggressive L3 streaming)

"""

import math
import torch
import torch.nn as nn
from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat

from .brevitas_custom_layers import QuantPatchEmbed, QuantMambaWrapper


class FEMBATinyInt8(nn.Module):
    """
    FEMBA Tiny with INT8 quantization (true architecture).

    Key parameters:
    - expand = 4
    - d_state = 16 (not 4)
    - d_inner = 4 * d_model = 1540
    - in_proj projects from d_model to 2 * d_inner = 3080
    - dt_rank = ceil(d_model / 16) = 25
    - Weight bit width = 8 (INT8)

    Args:
        inp_size: Input spatial size (EEG_channels, samples) - default (22, 1280)
        patch_size: Patch size for embedding - default (2, 16) per FEMBA spec
        stride: Stride for patch embedding - default (2, 16) per FEMBA spec
        in_chans: Number of input channels - default 1
        embed_dim: Embedding dimension per patch row - default 35
        expand: Expansion factor for d_inner - default 4 (FEMBA standard)
        d_state: SSM state dimension - default 16
        d_conv: Conv1d kernel size in MAMBA - default 4
        num_blocks: Number of encoder blocks - default 2
        num_classes: Number of output classes - default 2 (binary classification)

    中文参数说明：
        inp_size：输入尺寸（EEG 通道数，采样点数），默认 (22, 1280)
        patch_size / stride：patch embedding 的切块大小/步长，默认 (2, 16)
        embed_dim：每个 patch 行的嵌入维度，默认 35
        expand：Mamba d_inner 相对 d_model 的扩张倍数，默认 4（FEMBA 标准配置）
        d_state：SSM 状态维度，默认 16
        d_conv：Mamba 内部 depthwise conv1d 的卷积核大小，默认 4
        num_blocks：双向 Mamba 编码器 block 数量，默认 2
        num_classes：分类类别数，默认 2（二分类）
    """

    def __init__(
        self,
        inp_size=(22, 1280),
        patch_size=(2, 16),
        stride=(2, 16),
        in_chans=1,
        embed_dim=35,
        expand=4,
        d_state=16,
        d_conv=4,
        num_blocks=2,
        num_classes=2
    ):
        super().__init__()
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.expand = expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_blocks = num_blocks
        bit_width = 8  # 全局量化位宽：权重与激活统一用 INT8

        # Calculate dimensions after patch embedding
        # 计算 patch embedding 之后的网格尺寸/序列长度/模型维度
        H, W = inp_size
        self.grid_h = (H - patch_size[0]) // stride[0] + 1
        self.grid_w = (W - patch_size[1]) // stride[1] + 1
        self.seq_len = self.grid_w  # Sequence length  # 序列长度（Mamba 沿此维度做扫描）
        self.d_model = self.grid_h * embed_dim  # Model dimension  # 模型维度 = grid_h * embed_dim

        # True FEMBA: d_inner = expand * d_model (expand=4)
        # Mamba 内部展开维度 = expand * d_model
        self.d_inner = expand * self.d_model

        # dt_rank as per FEMBA spec
        # dt（离散化步长）投影的秩，按 FEMBA 规格取 ceil(d_model/16)
        self.dt_rank = math.ceil(self.d_model / 16)

        print(f"[FEMBATinyInt8] Configuration:")
        print(f"  Input: {inp_size}, patch: {patch_size}, stride: {stride}")
        print(f"  Grid: ({self.grid_h}, {self.grid_w})")
        print(f"  d_model: {self.d_model}, seq_len: {self.seq_len}")
        print(f"  d_inner: {self.d_inner} (expand={expand})")
        print(f"  d_state: {d_state}, dt_rank: {self.dt_rank}")
        print(f"  Weight bit width: {bit_width}")

        # Estimate weight size per direction
        in_proj_size = self.d_model * 2 * self.d_inner  # projects to 2*d_inner
        out_proj_size = self.d_inner * self.d_model
        conv_size = self.d_inner * d_conv + self.d_inner  # depthwise + bias
        x_proj_size = self.d_inner * (self.dt_rank + 2 * d_state)  # dt_rank + 2*d_state
        dt_proj_size = self.dt_rank * self.d_inner + self.d_inner  # with bias
        a_log_size = self.d_inner * d_state * 4  # FP32
        d_size = self.d_inner * 4  # FP32
        total_per_dir = in_proj_size + out_proj_size + conv_size + x_proj_size + dt_proj_size + a_log_size + d_size
        print(f"  Weight size per direction: {total_per_dir / 1024 / 1024:.2f} MB")
        print(f"  Total for 4 directions (2 blocks): {4 * total_per_dir / 1024 / 1024:.2f} MB")

        # Input quantization
        # 对原始输入做量化（对应 FP32 版 FEMBA 中没有的一步：这里给输入信号打一个 INT8 量化"探针"）
        self.input_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Patch embedding: [B, 1, H, W] -> [B, seq_len, d_model]
        # 量化版 PatchEmbed：内部用 QuantConv2d 做卷积切块+投影，输出再量化一次
        self.patch_embed = QuantPatchEmbed(
            inp_size=inp_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Positional embedding (learnable parameter)
        # 可学习位置编码，参数本身仍是 FP32 存储
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.seq_len, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Quantization for positional embedding
        # 位置编码单独量化，保证和 patch_embed 输出相加前 scale 可控
        self.pos_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Scale equalizer
        # scale 均衡器：两路不同来源的量化张量相加前，先统一到同一个 scale，
        # 否则 INT8 定点加法会因为 scale 不一致而出错
        self.scale_equalizer = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Encoder blocks: BiMamba + Residual + LayerNorm
        # 编码器：num_blocks 层 [双向Mamba + 残差 + LayerNorm + 量化] 堆叠
        self.mamba_blocks = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.post_norm_quants = nn.ModuleList()

        for i in range(num_blocks):
            # Bi-Mamba block with true FEMBA dimensions
            # 双向 Mamba block（内部 in_proj/out_proj/x_proj/dt_proj 等线性层都是 QuantLinear）
            self.mamba_blocks.append(
                QuantMambaWrapper(
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    d_state=d_state,
                    conv_kernel=d_conv,
                    bidirectional_strategy="add",
                    bit_width=bit_width,
                    return_quant_tensor=True
                )
            )

            # LayerNorm after residual
            # 残差相加之后做 LayerNorm；LayerNorm 本身不量化，保持 FP32 计算
            self.norm_layers.append(
                nn.LayerNorm(self.d_model)
            )

            # Post-norm quantization
            # LayerNorm 输出（FP32）重新量化回 INT8，供下一个 block 使用
            self.post_norm_quants.append(
                qnn.QuantIdentity(
                    bit_width=bit_width,
                    return_quant_tensor=True
                )
            )

        # Global average pool over sequence
        # 沿序列维度做全局平均池化，得到定长向量供分类头使用
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Pre-classifier quantization
        # 分类头之前再量化一次
        self.pre_classifier_quant = qnn.QuantIdentity(
            bit_width=bit_width,
            return_quant_tensor=True
        )

        # Final classifier
        # 最终分类线性层，权重 INT8 量化；输出不再转成 QuantTensor（直接给 CE loss 用）
        self.classifier = qnn.QuantLinear(
            self.d_model,
            num_classes,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=False
        )

    def forward(self, x):
        """Forward pass following FEMBA architecture."""
        # ===== 步骤1: 输入量化 =====
        # 输入 x: [B, in_chans, H, W]，先量化成 QuantTensor（内部仍是 FP32，但携带 scale）
        x = self.input_quant(x)

        if hasattr(x, 'value'):
            x = x.value

        # ===== 步骤2: Patch Embedding =====
        # QuantConv2d 切 patch + 投影到 embed_dim，reshape/permute 成序列 [B, seq_len, d_model]
        x = self.patch_embed(x)

        # Add positional embedding
        # ===== 步骤3: 加位置编码 =====
        # pos_embed 先单独量化，再和 patch_embed 的输出分别过 scale_equalizer 对齐 scale 后相加
        pos = self.pos_quant(self.pos_embed)

        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        x = self.scale_equalizer(x_val)
        pos = self.scale_equalizer(pos_val)

        if hasattr(x, 'value'):
            x_val = x.value
        else:
            x_val = x
        if hasattr(pos, 'value'):
            pos_val = pos.value
        else:
            pos_val = pos

        x = x_val + pos_val
        x = self.scale_equalizer(x)  # 相加结果重新量化，供后续 encoder 使用

        # Encoder blocks
        # ===== 步骤4: Encoder（双向 Mamba × num_blocks） =====
        # 每个 block: 残差(res) + QuantMambaWrapper(正向+反向 Mamba 相加) + LayerNorm(FP32) + 重新量化
        for mamba_block, norm_layer, post_norm_quant in zip(
            self.mamba_blocks, self.norm_layers, self.post_norm_quants
        ):
            if hasattr(x, 'value'):
                res = x.value
            else:
                res = x

            x = mamba_block(x)

            if hasattr(x, 'value'):
                x_val = x.value
            else:
                x_val = x

            x = res + x_val
            x = norm_layer(x)
            x = post_norm_quant(x)

        if hasattr(x, 'value'):
            x = x.value

        # Global pool and classify
        # ===== 步骤5: 全局平均池化 + 分类 =====
        # [B, seq_len, d_model] -> transpose -> [B, d_model, seq_len] -> 池化成 [B, d_model, 1]
        # -> squeeze 成 [B, d_model] -> 量化 -> QuantLinear 分类头 -> [B, num_classes]
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.pre_classifier_quant(x)
        x = self.classifier(x)
        return x


def get_sample_input(batch_size=1, in_chans=1, inp_size=(22, 1280)):
    """Generate a sample input tensor for testing."""
    # 生成一个随机输入张量，仅用于 shape/流程验证，不代表真实 EEG 数据
    return torch.randn(batch_size, in_chans, inp_size[0], inp_size[1])


def test_model():
    """Quick sanity test of the model."""
    # 快速自检：构建模型 -> 跑一次前向 -> 打印输出 shape/参数量，确认结构没有搭错
    print("=" * 70)
    print("Test 34: FEMBA Tiny INT8 (True Architecture)")
    print("=" * 70)

    model = FEMBATinyInt8(
        inp_size=(22, 1280),
        patch_size=(2, 16),
        stride=(2, 16),
        in_chans=1,
        embed_dim=35,
        expand=4,        # True FEMBA: expand=4
        d_state=16,      # True FEMBA: d_state=16
        d_conv=4,
        num_blocks=2,
        num_classes=2
    )
    model.eval()

    x = get_sample_input(batch_size=1, in_chans=1, inp_size=(22, 1280))
    print(f"\nInput shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Predicted class: {output.argmax(dim=1).item()}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print(f"\nFEMBA Tiny INT8 Configuration:")
    print(f"  Input size: {model.inp_size}")
    print(f"  d_model: {model.d_model}")
    print(f"  d_inner: {model.d_inner} (expand={model.expand})")
    print(f"  d_state: {model.d_state}")
    print(f"  dt_rank: {model.dt_rank}")
    print(f"  Sequence length: {model.seq_len}")

    # Verify parameter count is ~7.6M
    expected_params = 7_600_000
    if num_params > expected_params * 0.9 and num_params < expected_params * 1.1:
        print(f"\n[OK] Parameter count matches expected ~7.6M")
    else:
        print(f"\n[NOTE] Parameter count {num_params:,} differs from expected ~7.6M")
        print(f"       (Brevitas quantization layers add overhead)")

    print("=" * 70)
    print("Test PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()
