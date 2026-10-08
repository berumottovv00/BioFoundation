# 第 5、6 步：QAP 配置 + 假数据冒烟训练

目标：用假数据把 **QAP 预训练的完整流程**（Hydra 配置 → DataModule → MaskTask → INT8 模型 → Trainer → checkpoint）从头到尾跑一遍，确认各环节都能连起来。
这一步不看模型学得好不好（假数据是白噪声，loss 基本不会降），只看**能不能跑完、数值正常、文件正常保存**。

所有命令除特别说明外，都在 AutoDL 服务器上运行。每一步都给了两种写法时：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，`EOF` 会失效。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 改动

### 第 5 步：新建 `config/model/FEMBA_tiny_int8_pretrain.yaml`

```yaml
model:
  _target_: models.FEMBA_int8.FEMBATinyInt8Pretrain   # QAP 模型：INT8 编码器 + FP32 Decoder
  seq_length: 1280                                     # 每段 1280 个采样点（256 Hz × 5 秒）
  num_channels: 22                                     # 22 个 EEG 通道
  exp: 4                                               # Mamba 内部扩张倍数
  num_blocks: 2                                        # 2 层双向 Mamba
  embed_dim: 35                                        # patch 嵌入维度
```

跟 FP 版的 `FEMBA_tiny_pretrain.yaml` 相比：只换了 `_target_`，去掉了 `num_classes: 0`（QAP 模型只有重建分支，不需要这个参数）。

### 第 6 步：新建 `config/experiment/FEMBA_qap_demo.yaml`

复制自 `FEMBA_pretrain_demo.yaml`，只改了三处：

| 项 | `FEMBA_pretrain_demo.yaml`（FP 版） | `FEMBA_qap_demo.yaml`（QAP） | 原因 |
|---|---|---|---|
| `model` | `FEMBA_tiny_pretrain` | `FEMBA_tiny_int8_pretrain` | 换成 QAP 模型 |
| `batch_size` | 8 | 32 | 更接近真实训练；4090 24G 足够 |
| `io.base_output_path` | 没设（继承默认的 `#CHANGEME`） | `${env:CHECKPOINT_DIR}/tb_logs` | 否则 TensorBoard 日志会写进一个叫 `#CHANGEME` 的目录 |

其余保持不变：1 个 epoch、训练最多 20 个 batch、验证最多 5 个 batch、AdamW lr 1e-3、按 `val_loss` 保存最优 checkpoint。

### 本地已做的检查

在 Mac 上用 Hydra 组合配置并解析所有变量（没创建模型，本地没装 brevitas），结果：

```
model: {'_target_': 'models.FEMBA_int8.FEMBATinyInt8Pretrain', 'seq_length': 1280, 'num_channels': 22, 'exp': 4, 'num_blocks': 2, 'embed_dim': 35}
batch_size: 32 | num_workers: 2 | optimizer: {'optim': 'AdamW', 'lr': 0.001}
io: {'base_output_path': '/root/autodl-tmp/experiments/tb_logs', 'checkpoint_dirpath': '/root/autodl-tmp/experiments/checkpoints', ...}
hdf5: /root/autodl-tmp/data/demo/pretrain_fake.h5
trainer: {'accelerator': 'gpu', 'devices': 1, 'max_epochs': 1, 'limit_train_batches': 20, 'limit_val_batches': 5, 'num_sanity_val_steps': 0}
```

---

## 1. 同步文件到服务器（在 **Mac** 上运行）

`端口号` 和 `主机` 换成 AutoDL 控制台"登录指令"里的值（例如 `ssh -p 12345 root@connect.westb.seetacloud.com` → 端口 `12345`，主机 `connect.westb.seetacloud.com`）。

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 端口号" \
  config/model/FEMBA_tiny_int8_pretrain.yaml \
  config/experiment/FEMBA_qap_demo.yaml \
  config/data_module/pretrain_data_module_demo.yaml \
  make_datasets/make_fake_pretrain_demo.py \
  models/FEMBA_int8.py models/femba_decoder.py \
  tasks/pretrain_task.py \
  root@主机:/root/autodl-tmp/BioFoundation/
```

后面几个文件之前可能已经传过，再传一次没关系（内容相同时 rsync 会跳过）。

---

## 2. 环境检查（服务器）

确认训练要用到的包都能导入，特别是 `pytorch_lightning`（会顺带导入 scipy，之前出过 numpy 残留问题）：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch, brevitas, scipy.special, pytorch_lightning, hydra, h5py; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望：`torch 2.7.1+cu126 | cuda True NVIDIA GeForce RTX 4090`。
如果报 `All ufuncs must have type numpy.ufunc`，按 `docs/02_qap_step4_verification.md` 最后的方法修。

---

## 3. 设置环境变量 + 生成假数据（服务器）

```bash
cd /root/autodl-tmp/BioFoundation

export DATA_PATH=/root/autodl-tmp/data                 # 数据根目录；配置里的 ${env:DATA_PATH} 指向这里
export CHECKPOINT_DIR=/root/autodl-tmp/experiments     # 输出根目录：checkpoint、TensorBoard 日志、Hydra 运行目录都在这下面
mkdir -p $DATA_PATH $CHECKPOINT_DIR

# 生成 1000 个假样本（每个 22 通道 × 1280 点的高斯白噪声），约 113 MB
python make_datasets/make_fake_pretrain_demo.py --out $DATA_PATH/demo/pretrain_fake.h5 --num-samples 1000

ls -lh $DATA_PATH/demo/                                # 应该看到 pretrain_fake.h5，约 108M
```

**为什么是 1000 个样本**：按 8:2 划分，训练 800 个、验证 200 个。batch 32 时训练有 25 个 batch（被 `limit_train_batches` 限制到 20 个），
验证有 6 个 batch（限制到 5 个）。如果用默认的 300 个，训练只有 7 个 batch、验证只有 1 个，测试覆盖太少。

`export` 只对当前终端有效，换终端或重新登录后要重新执行这两行。

---

## 4. 训练前先检查配置（服务器）

只打印 Hydra 组合出来的最终配置，不开始训练：

```bash
cd /root/autodl-tmp/BioFoundation && python run_train.py +experiment=FEMBA_qap_demo --cfg job --resolve | grep -A7 "^model:"
```

期望看到 `_target_: models.FEMBA_int8.FEMBATinyInt8Pretrain` 和那几个参数。

---

## 5. 跑冒烟训练（服务器）

```bash
cd /root/autodl-tmp/BioFoundation
python -u run_train.py +experiment=FEMBA_qap_demo 2>&1 | tee /root/autodl-tmp/qap_demo.log
# -u：不缓冲输出，日志实时显示
# 2>&1 | tee ...：屏幕上正常显示，同时把完整输出存一份到 qap_demo.log，方便事后查看
```

### 正常运行时会依次看到

| 顺序 | 输出 | 说明 |
|---|---|---|
| 1 | 一大段 YAML | 最终配置 |
| 2 | `===> Loading datasets`、`800 200` | 训练集 800 个、验证集 200 个 |
| 3 | `===> Start building model`、`[FEMBATinyInt8] Configuration` | 创建 QAP 模型 |
| 4 | `No pretrained checkpoint provided` | 从零开始训练，正常 |
| 5 | 模型结构表（ModelSummary） | 总参数量约 760 万 |
| 6 | 进度条 `Epoch 0: ... 20/20 ... train_loss_step=...` | 训练 20 步 |
| 7 | 验证进度条、`val_loss=...` | 验证 5 个 batch |
| 8 | `Best checkpoint path: .../checkpoints/FEMBA_qap_demo/<版本>/...ckpt` | 最优 checkpoint 已保存 |
| 9 | 又一次验证（`final_validate`）| 用最优 checkpoint 重新验证一次，打印 `val_loss` 表格 |

### 判断通过的标准

- 整个过程跑完，**没有 Traceback**
- `train_loss` 和 `val_loss` 都是正常的有限数字，**不是 `nan` 或 `inf`**
- 打印出了 `Best checkpoint path`
- loss **不下降是正常的**：输入是白噪声，被遮住的部分本来就没法预测

可以忽略的输出：`Named tensors ... experimental`（Brevitas）、`Hydra14MigrationWarning`、DataLoader worker 数量相关的提示。

---

## 6. 训练后检查输出文件（服务器）

```bash
ls -R $CHECKPOINT_DIR/checkpoints/FEMBA_qap_demo/ | head        # 应该有 last.ckpt 和一个按 val_loss 命名的最优 ckpt
ls $CHECKPOINT_DIR/tb_logs/FEMBA_qap_demo/                       # TensorBoard 日志目录
grep -n -i "error\|nan\|Traceback" /root/autodl-tmp/qap_demo.log | head   # 期望什么都不输出
```

---

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `CUDA out of memory` | 量化版 Mamba 是逐时间步循环计算，显存占用比 FP 版大。命令末尾加 `batch_size=16` 覆盖：`python -u run_train.py +experiment=FEMBA_qap_demo batch_size=16` |
| `All ufuncs must have type numpy.ufunc` | numpy 残留文件，见 `docs/02_qap_step4_verification.md` |
| `Missing environment variable DATA_PATH` 之类 | 当前终端没执行第 3 步的两行 `export` |
| `IndexError: index 16 is out of bounds` | `tasks/pretrain_task.py` 没同步到服务器（第 4 步的修复） |
| `FileNotFoundError ... pretrain_fake.h5` | 没生成假数据，或 `DATA_PATH` 跟生成时不一致 |
| `No module named 'models.FEMBA_int8'` | 模型文件没同步，或不是在项目根目录运行 |

---

## 运行中遇到的问题

### 1. `Could not find 'model/FEMBA_tiny_int8_pretrain'`

服务器上缺少新建的配置文件。rsync 要在 **Mac 终端**里运行（提示符是 `shiyu@... %`，不是 `root@autodl-container-...#`），重新同步后解决。

### 2. `MultiStepLR.__init__() takes from 3 to 5 positional arguments but 6 were given`

**仓库原有 bug，跟 QAP 无关。** `schedulers/multi_step_lr.py` 调用父类时把 `verbose` 作为第 5 个位置参数传进去：

```python
super(MultiStepLRWarmup, self).__init__(optimizer, milestones, gamma, last_epoch, verbose)
```

torch 2.7 的 `MultiStepLR` 签名是 `(optimizer, milestones, gamma=0.1, last_epoch=-1)`，`verbose` 在 2.2 起弃用、2.7 已删除。
项目 `uv.lock` 锁定的就是 torch 2.7.1，所以原版 FP 预训练 `FEMBA_pretrain.yaml` 也会碰到这个错。

**修复**：删掉 `MultiStepLRWarmup.__init__` 的 `verbose` 参数，调用父类时不再传。`verbose` 默认是 `False`，唯一的调用方 `multi_step_lr()` 也从没传过它，所以行为不变。

**验证**（Mac 本地，torch 2.7.1）：`milestones='3+6', gamma=0.1` 时学习率为 `1e-3 ×3 → 1e-4 ×3 → 1e-5`，正确。

**已知的另一个原有问题（未修改）**：开启 warmup（`warmup_iter > 0`）时，warmup 结束那一步的学习率是在 warmup 最后一步的值上乘 gamma，
而不是在基础学习率上乘（例：`warmup_iter=4, milestones=[4], gamma=0.5` 时得到 `3.75e-4`，期望 `5e-4`）。
现有配置都是 `warmup_iter: -1`，不会触发，暂不处理。

### 3. `UnpicklingError: Weights only load failed ... omegaconf.dictconfig.DictConfig`

**训练本身已经跑通**（20 步、`train_loss_epoch=0.154`、`val_loss_epoch=0.128`、checkpoint 已保存），报错发生在训练后 `final_validate` 加载最优 checkpoint 时。

**仓库原有问题，跟 QAP 无关。** `MaskTask` 用 `save_hyperparameters(hparams)` 把 Hydra 的 `DictConfig` 存进了 checkpoint；
torch 2.6 起 `torch.load` 默认 `weights_only=True`，只允许张量等基本类型，遇到 `DictConfig` 就拒绝加载。

**修复**：`run_train.py` 里三处加载 checkpoint 的调用都加上 `weights_only=False`：

| 调用 | 用途 |
|---|---|
| `trainer.fit(..., ckpt_path=last_ckpt, weights_only=False)` | 断点续训 |
| `trainer.validate(..., ckpt_path=best_ckpt, weights_only=False)` | 训练后用最优 checkpoint 再验证（本次报错处） |
| `trainer.test(..., ckpt_path=ckpt, weights_only=False)` | 最终测试 |

加载的都是本次训练自己生成的 checkpoint，来源可信；仓库里 `tasks/finetune_task_EMG.py`、`tasks/finetune_regression_task_LuMamba.py` 也是这么写的。

**验证**（Mac 本地，torch 2.7.1 + Lightning 2.6.6，用一个存了 `DictConfig` 超参数的小模型）：
不传 `weights_only` → `UnpicklingError`（复现）；`weights_only=False` → 正常加载。

**以后会遇到的同类问题（未修改）**：`tasks/finetune_task.py` 的 `load_pretrained_checkpoint` 用 `torch.load(model_ckpt)` 加载预训练权重，
做 QAP 权重的量化微调时会报同样的错，到时一起处理。

---

## 结果（2026-09-27，RTX 4090）✅ 全流程跑通

| 项 | 结果 |
|---|---|
| 模型参数 | 8.7 M 可训练；4 个不可训练 = 4 个 `conv1d.weight_scale` 占位参数（跟 `docs/01_qap_verification.md` 第 3 步一致） |
| Decoder | 2.1 K 参数（FP32） |
| 训练 | 20 步，约 19 秒（≈1 步/秒，batch 32） |
| `train_loss_epoch` | 0.154 |
| `val_loss_epoch`（训练中） | 0.128 |
| 最优 checkpoint | `.../checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_27_09_05-31-02.549489/epoch=0-step=20.ckpt` |
| `final_validate`（加载最优 checkpoint 后重新验证） | `val_loss_epoch = 0.1283`，跟训练中的 0.1284 基本一致（每次验证随机生成的掩码不同，有微小差异） |

没有 Traceback，loss 全程是正常的有限值，checkpoint 能保存也能重新加载。
**QAP 预训练流程（Hydra 配置 → DataModule → MaskTask → INT8 模型 → Trainer → checkpoint 保存/加载）全部打通。**
