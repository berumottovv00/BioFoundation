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
#*----------------------------------------------------------------------------*
"""
FEMBA Tiny INT8 的 QAP（Quantization-Aware Pretraining，量化感知预训练）与量化微调模型。

两个类都继承 ARES/tests/test_networks/test_25_femba_tiny_int8.py 里的 FEMBATinyInt8（不修改原文件），
所以 INT8 编码器的参数名完全相同，预训练权重可以直接加载到微调模型：

- FEMBATinyInt8Pretrain：去掉分类头，接上 FP32 的 Decoder 做掩码重建。
  接口跟 models/FEMBA.py 的 FEMBA(num_classes=0) 一致：forward(x, mask) -> (重建信号, 原始信号)，
  供 tasks/pretrain_task.py 的 MaskTask 使用。
- FEMBATinyInt8Classifier：保留原版 INT8 分类头，只做接口适配。
  接口跟 FEMBA(num_classes>0) 一致：forward(x, mask) -> (logits, 原始信号)，
  供 tasks/finetune_task.py 的 FinetuneTask 使用。

不依赖 mamba_ssm：量化版 Mamba（QuantSSM）是纯 PyTorch 实现。
"""

from typing import Optional, Tuple

import torch

from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8
from models.femba_decoder import Decoder


class FEMBATinyInt8Pretrain(FEMBATinyInt8):
    """
    INT8 FEMBA encoder + FP32 reconstruction decoder for quantization-aware pretraining.

    Encoder submodules are created by FEMBATinyInt8.__init__, so parameter names are
    identical to FEMBATinyInt8 and pretrained weights load into it directly for
    quantized fine-tuning (the classifier head is then freshly initialised).

    Args:
        seq_length (int): Number of time samples per window (W).
        num_channels (int): Number of EEG channels (H).
        embed_dim (int): Patch embedding dimension.
        num_blocks (int): Number of bidirectional Mamba blocks.
        exp (int): Mamba expansion factor (d_inner = exp * d_model).
        d_state (int): SSM state dimension.
        d_conv (int): Mamba depthwise conv kernel size.
        patch_size (Tuple[int, int]): Patch size.
        stride (Tuple[int, int]): Patch stride.
        kernel_dec (Tuple[int, int]): Decoder smoothing conv kernel size.

    中文说明：
    编码器部分（输入量化、PatchEmbed、位置编码、双向 Mamba、LayerNorm 后的重新量化）
    全部是 INT8 伪量化，预训练时就让编码器适应量化误差；Decoder 只在预训练时使用、
    部署时丢弃，所以保持 FP32。
    """
    def __init__(self,
                 seq_length: int = 1280,
                 num_channels: int = 22,
                 embed_dim: int = 35,
                 num_blocks: int = 2,
                 exp: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 patch_size: Tuple[int, int] = (2, 16),
                 stride: Tuple[int, int] = (2, 16),
                 kernel_dec: Tuple[int, int] = (31, 31)):
        # Hydra 传进来的是 ListConfig，QuantPatchEmbed 用 isinstance(..., tuple) 判断，必须先转成 tuple
        patch_size = tuple(patch_size)
        stride = tuple(stride)
        kernel_dec = tuple(kernel_dec)

        super().__init__(
            inp_size=(num_channels, seq_length),
            patch_size=patch_size,
            stride=stride,
            in_chans=1,
            embed_dim=embed_dim,
            expand=exp,
            d_state=d_state,
            d_conv=d_conv,
            num_blocks=num_blocks,
        )

        # 预训练不需要分类头，删掉，避免出现不参与训练的参数
        del self.global_pool
        del self.pre_classifier_quant
        del self.classifier

        self.decoder = Decoder(
            embed_dim=embed_dim,
            grid_size=(self.grid_h, self.grid_w),
            kernel_dec=kernel_dec,
            patch_size=patch_size,
            stride=stride,
        )

    def forward_features(self, x):
        """
        INT8 encoder: [B, 1, H, W] -> [B, seq_len, d_model] (plain tensor, before pooling).

        注意：这是 FEMBATinyInt8.forward 里步骤1~4 的逐行拷贝。
        如果 test_25_femba_tiny_int8.py 里的 forward 改了，这里必须同步修改，
        否则预训练时的量化计算会和部署时不一致。
        """
        # ===== 步骤1: 输入量化 =====
        x = self.input_quant(x)

        if hasattr(x, 'value'):
            x = x.value

        # ===== 步骤2: Patch Embedding =====
        x = self.patch_embed(x)

        # ===== 步骤3: 加位置编码 =====
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
        x = self.scale_equalizer(x)

        # ===== 步骤4: Encoder（双向 Mamba × num_blocks） =====
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

        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input signal of shape (B, C, T).
            mask (torch.BoolTensor): Mask of shape (B, C, T), True where masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (reconstructed (B, C, T), original input (B, C, T)).
        """
        # 跟 FEMBA.forward 一样：被掩码的区域置零，原始信号留着算重建损失
        x_original = x
        x_masked = x.clone()
        x_masked[mask] = 0

        # FEMBATinyInt8 的输入是 [B, 1, H, W]
        features = self.forward_features(x_masked.unsqueeze(1))  # (B, seq_len, d_model)
        x_reconstructed = self.decoder(features)  # (B, C, T)
        return x_reconstructed, x_original


class FEMBATinyInt8Classifier(FEMBATinyInt8):
    """
    INT8 FEMBA classifier with the FinetuneTask interface, for quantized fine-tuning.

    Only adapts the interface of FEMBATinyInt8; the computation (INT8 encoder, global
    average pooling, INT8 linear head) is the parent's forward, unchanged. Parameter
    names match FEMBATinyInt8Pretrain's encoder, so QAP-pretrained weights load directly.

    Args:
        seq_length (int): Number of time samples per window (W).
        num_channels (int): Number of EEG channels (H).
        embed_dim (int): Patch embedding dimension.
        num_blocks (int): Number of bidirectional Mamba blocks.
        exp (int): Mamba expansion factor (d_inner = exp * d_model).
        d_state (int): SSM state dimension.
        d_conv (int): Mamba depthwise conv kernel size.
        patch_size (Tuple[int, int]): Patch size.
        stride (Tuple[int, int]): Patch stride.
        num_classes (int): Number of output classes.
        classification_type (str): "bc", "mcc" or "ml". "mc"/"mmc" need a per-channel
            head that FEMBATinyInt8 does not have.

    中文说明：
    FinetuneTask 调用 self.model(X, mask) 并期望返回 (logits, _)，而 FEMBATinyInt8 是
    forward(x)、输入 [B, 1, H, W]、只返回 logits。这个类只负责这层转换。
    分类头是"全局平均池化 + 一个 INT8 线性层"，比 FP 版 FEMBA 的 MambaClassifier 简单，
    所以只支持每个样本输出一个类别的任务（bc/mcc/ml）。
    """
    SUPPORTED_CLASSIFICATION_TYPES = ("bc", "mcc", "ml")

    def __init__(self,
                 seq_length: int = 1280,
                 num_channels: int = 22,
                 embed_dim: int = 35,
                 num_blocks: int = 2,
                 exp: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 patch_size: Tuple[int, int] = (2, 16),
                 stride: Tuple[int, int] = (2, 16),
                 num_classes: int = 2,
                 classification_type: str = "bc"):
        if classification_type not in self.SUPPORTED_CLASSIFICATION_TYPES:
            raise NotImplementedError(
                f"classification_type={classification_type!r} is not supported by the INT8 head; "
                f"use one of {self.SUPPORTED_CLASSIFICATION_TYPES}"
            )
        # Hydra 传进来的是 ListConfig，QuantPatchEmbed 用 isinstance(..., tuple) 判断，必须先转成 tuple
        patch_size = tuple(patch_size)
        stride = tuple(stride)

        super().__init__(
            inp_size=(num_channels, seq_length),
            patch_size=patch_size,
            stride=stride,
            in_chans=1,
            embed_dim=embed_dim,
            expand=exp,
            d_state=d_state,
            d_conv=d_conv,
            num_blocks=num_blocks,
            num_classes=num_classes,
        )
        self.classification_type = classification_type

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input signal of shape (B, C, T).
            mask (torch.BoolTensor, optional): Ignored. FinetuneTask always passes an
                all-False mask; accepted only to match the FEMBA interface.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (logits (B, num_classes), original input (B, C, T)).
        """
        logits = super().forward(x.unsqueeze(1))  # FEMBATinyInt8 的输入是 [B, 1, H, W]
        return logits, x
