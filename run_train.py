# *----------------------------------------------------------------------------*
# * Copyright (C) 2025 ETH Zurich, Switzerland                                 *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Thorir Mar Ingolfsson                                             *
# * Author:  Anna Tegon                                                        *
# * Author:  Berkay Döner                                                      *
# * Author:  Matteo Fasulo                                                     *
# *----------------------------------------------------------------------------*
"""
训练入口脚本。命令行用法示例：
    python run_train.py +experiment=FEMBA_pretrain_demo

整体流程：加载 Hydra 配置 -> 实例化 DataModule -> 实例化 LightningModule（内部会
再实例化具体模型，见 tasks/pretrain_task.py 或 tasks/finetune_task.py）->
（可选）加载预训练权重 -> 配置 Trainer/回调 -> 训练 -> 验证 -> 测试。
"""
import logging
import os
import os.path as osp
from datetime import datetime
from logging import Logger

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb


from biofoundation.core.environment import require_environment
from util.train_utils import find_last_checkpoint_path

OmegaConf.register_new_resolver("env", lambda key: os.getenv(key))
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

logger: Logger = logging.getLogger(__name__)

# Set float32 matmul precision to high for better performance on supported hardware
torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig):
    # 固定随机种子，保证实验可复现
    seed_everything(cfg.seed)

    date_format = "%d_%m_%H-%M-%S.%f"

    # 用 "tag + 时间戳" 拼出这次运行的版本名，用于区分 checkpoint/日志目录
    version = f"{cfg.tag}_{datetime.now().strftime(date_format)}"

    # tensorboard 日志（一直启用）
    tb_logger = TensorBoardLogger(
        save_dir=osp.expanduser(cfg.io.base_output_path), name=cfg.tag, version=version
    )

    loggers = [tb_logger]

    # Weights & Biases（可选：只有配置里写了 wandb 字段才会启用）
    wandb_cfg = cfg.get("wandb", None)
    wandb_logger = None
    if wandb_cfg:
        wandb_logger = WandbLogger(
            entity=wandb_cfg.entity,
            project=wandb_cfg.project,
            save_dir=wandb_cfg.save_dir,
            name=wandb_cfg.run_name if wandb_cfg.run_name else version,
            offline=wandb_cfg.offline,
        )
        loggers.append(wandb_logger)

    # ===== 步骤1: 实例化 DataModule =====
    # 根据配置里 data_module._target_（比如 PretrainDataModule/FinetuneDataModule）动态构建
    print("===> Loading datasets")
    data_module = hydra.utils.instantiate(cfg.data_module)

    # ===== 步骤2: 实例化 LightningModule（内部会再实例化具体的模型，见 MaskTask/FinetuneTask） =====
    # 注意这里传的是 cfg.task（比如 pretrain_task/finetune_task 的配置）+ 整个 cfg 本身（hparams 都保存进去）
    print("===> Start building model")
    model = hydra.utils.instantiate(cfg.task, cfg)
    print(model)

    safetensors_path = cfg.get("pretrained_safetensors_path", None)
    checkpoint_path = cfg.get("pretrained_checkpoint_path", None)

    # ===== 步骤3: （可选）加载预训练权重，用于微调场景 =====
    if safetensors_path is not None:
        print(f"===> Loading pretrained safetensors from {safetensors_path}")
        # Assuming your model has this method
        model.load_safetensors_checkpoint(safetensors_path)
    elif checkpoint_path is not None:
        print(f"===> Loading pretrained checkpoint from {checkpoint_path}")
        model.load_pretrained_checkpoint(checkpoint_path)
    else:
        print("No pretrained checkpoint provided. Proceeding without loading.")

    # 本次运行的 checkpoint 保存目录：<checkpoint_dirpath>/<tag>/<version>/
    checkpoint_dirpath = cfg.io.checkpoint_dirpath
    checkpoint_dirpath = osp.join(checkpoint_dirpath, cfg.tag, version)
    print(f"Checkpoint path: {checkpoint_dirpath}")
    last_ckpt = None
    if cfg.resume:
        # 断点续训：找这个目录下最近一次保存的 checkpoint
        last_ckpt = find_last_checkpoint_path(checkpoint_dirpath)
        print(f"last_ckpt_{last_ckpt}")
    print("===> Checkpoint callbacks")
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dirpath, **cfg.model_checkpoint
    )
    model_summary = pl.callbacks.ModelSummary(max_depth=4)
    callbacks = [model_checkpoint, model_summary]

    # 配置文件里额外声明的其他 Lightning 回调（比如 EarlyStopping 之类）
    print("===> Instantiate other callbacks")
    for _, callback in cfg.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback))

    # ===== 步骤4: 构建 Trainer =====
    # strategy=="ddp" 时要单独构造 DDPStrategy 对象（find_unused_parameters 这个参数
    # PyTorch-Lightning 默认是 True，Trainer 的 **cfg.trainer 展开不支持嵌套构造，所以手动处理）
    print("===> Instantiate trainer")
    if cfg.trainer.strategy == "ddp":
        del cfg.trainer.strategy
        trainer = Trainer(
            **cfg.trainer,
            logger=loggers,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=cfg.find_unused_parameters),
        )
    else:
        trainer = Trainer(
            **cfg.trainer,
            logger=loggers,
            callbacks=callbacks,
        )

    # ===== 步骤5: 训练 =====
    results: dict = {}
    if cfg.training:
        print("===> Start training")
        # weights_only=False：checkpoint 里存了 Hydra 的 DictConfig（save_hyperparameters），
        # torch>=2.6 默认 weights_only=True 会拒绝加载；这里只加载本次训练自己生成的 checkpoint，来源可信
        trainer.fit(model, data_module, ckpt_path=last_ckpt, weights_only=False)

    best_ckpt = model_checkpoint.best_model_path

    print(f"Best checkpoint path: {best_ckpt}")
    print(f"Best model score: {model_checkpoint.best_model_score}")

    # ===== 步骤6: 用最优 checkpoint 做一次验证 =====
    if cfg.final_validate:
        print("===> Start validation")
        trainer.validate(model, data_module, ckpt_path=best_ckpt, weights_only=False)

    # ===== 步骤7: 用最优 checkpoint 做一次测试 =====
    if cfg.final_test:
        # rank 0 only
        # Validate and test run on 1 device only (i.e. no distributed data parallelism)
        # This is to ensure reproducibility of metrics reported.
        # 多卡训练时先销毁进程组，测试阶段只用单卡跑，保证指标可复现（不受分布式聚合方式影响）

        del data_module, trainer
        print("Destroying process group...")
        if dist.is_initialized():
            dist.destroy_process_group()
        print("Destroyed process group.")

        if pl.utilities.rank_zero_only.rank == 0:
            print("Re-instantiating LightningDataModule for evaluation...")
            data_module = hydra.utils.instantiate(cfg.data_module)
            results, trainer = _run_test(
                module=model,
                datamodule=data_module,
                results=results,
                accelerator=cfg.trainer.accelerator,
                ckpt=best_ckpt,
                wandb_logger=wandb_logger,
            )

    # 如果根本没训练（纯评估模式），把当前模型状态另存一份 last.ckpt
    if not cfg.training:
        trainer.save_checkpoint(f"{checkpoint_dirpath}/last.ckpt")

    if wandb.run is not None:
        wandb.finish()


@pl.utilities.rank_zero_only
def _run_test(
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    results,
    accelerator,
    ckpt,
    wandb_logger=None,
):
    # 单卡（devices=1）重新起一个 Trainer 专门跑测试，避免多卡分布式指标聚合带来的偏差
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=wandb_logger if wandb_logger else [],
    )
    print("===> Start testing")
    test_results = trainer.test(module, datamodule=datamodule, ckpt_path=ckpt, weights_only=False)
    results["test_metrics"] = test_results
    return results, trainer


@hydra.main(config_path="./config", config_name="defaults", version_base="1.1")
def run(cfg: DictConfig):
    # Hydra 入口：cfg 是根据 config/defaults.yaml + 命令行传的 +experiment=xxx 合并出来的最终配置
    print(f"PyTorch-Lightning Version: {pl.__version__}")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    # 跑之前先检查两个必须的环境变量是否已设置（数据路径、checkpoint 保存路径）
    require_environment(("DATA_PATH", "CHECKPOINT_DIR"))
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run()
