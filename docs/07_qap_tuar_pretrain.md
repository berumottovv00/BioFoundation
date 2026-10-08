# 用 TUAR train.h5 做 QAP 预训练

目标：第一次用**真实 EEG** 做 QAP（INT8 量化感知预训练），得到一个真正学过 EEG 的 INT8 编码器，
后面再拿它做量化微调，跟"不预训练"的基线比较，初步判断 QAP 有没有用。

所有命令除特别说明外，都在 AutoDL 服务器上运行。每一步都给了两种写法：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，可能会出错。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 实验设计

| 项 | 取值 | 理由 |
|---|---|---|
| 数据 | TUAR `train.h5` 的 `X`（约 5.5 万段 × 22 通道 × 1280 点），**不读标签** | `HDF5Loader(finetune=False)` 只读 `X` |
| 不用 val.h5 / test.h5 | — | 它们留给量化微调做验证和测试，预训练碰了就是数据泄漏 |
| 预训练的训练/验证划分 | `train.h5` 随机 9:1（约 5 万 / 5.5 千） | 预训练的验证集只用来监控重建 loss |
| 模型 | `FEMBATinyInt8Pretrain`（INT8 编码器 + FP32 Decoder） | 跟冒烟测试相同 |
| 掩码 | 按 2×16 的块随机遮 60% | 跟 FEMBA 原版预训练相同 |
| batch | 32 | 冒烟测试在 4090 上跑过 |
| 优化器 | AdamW，lr 1e-4 | FEMBA 文档预训练用 lr 1e-4（冒烟测试的 1e-3 是随手设的） |
| 学习率调度 | `multi_step_lr`，里程碑 20 万 / 40 万步 → **本次等于恒定学习率** | `MaskTask` 创建调度器时只传优化器，cosine 还要总步数，接不上 |
| 轮数 | 10 个 epoch | 按 ~1 步/秒估算约 4～5 小时；先试跑测速再定 |
| checkpoint | 按 `val_loss` 保存最优 1 个 + `last.ckpt` | |

**局限**：数据量比 FEMBA 论文用的 TUEG 小得多，而且跟下游任务是同一个数据集（只是不同划分）。这是初步实验，结论不能直接推广。

---

## 改动

### 新建 `config/data_module/pretrain_data_module_tuar.yaml`

```yaml
data_module:
  _target_: 'data_module.pretrain_data_module.PretrainDataModule'
  name: "eeg_tuar_pretrain"
  cfg: {num_workers: ${num_workers}, batch_size: ${batch_size}}
  test: null
  train_val_split_ratio: 0.9
  datasets:
    tuar_train:
      _target_: 'datasets.hdf5_dataset.HDF5Loader'
      finetune: False                                   # 只读 X，忽略 y
      hdf5_file: '${env:DATA_PATH}/TUAR_data/train.h5'  # 只用 train 划分
```

### 新建 `config/experiment/FEMBA_qap_tuar.yaml`

基于 `FEMBA_qap_demo.yaml`，改动：

| 项 | `FEMBA_qap_demo`（冒烟） | `FEMBA_qap_tuar`（本次） |
|---|---|---|
| 数据 | 白噪声假数据 | TUAR `train.h5` |
| `num_workers` | 2 | 4 |
| `max_epochs` | 1 | 10 |
| `limit_train_batches` / `limit_val_batches` | 20 / 5 | 不限制（全量） |
| `log_every_n_steps` | 5 | 50 |
| lr | 1e-3 | 1e-4 |

---

## 本地已做的检查（Mac）

1. Hydra 组合 `+experiment=FEMBA_qap_tuar` 并解析：✅

   ```
   model: models.FEMBA_int8.FEMBATinyInt8Pretrain | task: tasks.pretrain_task.MaskTask | optimizer: {'optim': 'AdamW', 'lr': 0.0001}
   scheduler: {'_target_': 'schedulers.multi_step_lr.multi_step_lr', 'milestones': '200000+400000', 'gamma': 0.1, ...}
   trainer: {'devices': 1, 'max_epochs': 10, 'strategy': 'auto', 'log_every_n_steps': 50}
   io: {'base_output_path': '/root/autodl-tmp/experiments/tb_logs', 'checkpoint_dirpath': '/root/autodl-tmp/experiments/checkpoints', ...}
   ```

2. 用真实 TUAR `test.h5` 顶替 `train.h5`（格式完全相同），实际创建 DataModule 并取一个 batch：✅

   ```
   train/val sizes: 6489 722            （7211 个样本按 9:1 划分）
   sample type: Tensor | shape: (22, 1280) | dtype: torch.float32
   batch shape: (32, 22, 1280) | dtype: torch.float32 | finite: True
   ```

   `finetune=False` 时只返回 `X`（不是 `(X, y)` 元组），正是 `MaskTask` 需要的格式；float64 自动转成 float32。

   本地第一次用 `num_workers=0` 测试时报 `persistent_workers option needs num_workers > 0`：仓库的 DataLoader 固定开了
   `persistent_workers=True`，要求 `num_workers > 0`。服务器配置是 4，不受影响。

---

## 0. 换主机后先检查环境（服务器）

如果这次开机换了主机（主机名变了），先跑这一行：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch, scipy.special, pytorch_lightning, brevitas; print('ok', torch.__version__, 'nccl', torch.cuda.nccl.version(), '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望 `ok 2.7.1+cu126 nccl (2, 26, 2) | cuda True NVIDIA GeForce RTX 4090`。
报 `All ufuncs must have type numpy.ufunc` → 按 `docs/06_qap_finetune_step3_tuar_smoke.md` 的"换主机后的环境修复"处理。

---

## 1. 同步到服务器（在 **Mac** 终端运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 37878" \
  config/data_module/pretrain_data_module_tuar.yaml \
  config/experiment/FEMBA_qap_tuar.yaml \
  docs/07_qap_tuar_pretrain.md \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/BioFoundation/
```

---

## 2. 试跑：测速度和显存（服务器，几分钟）

正式跑之前，先用同样的配置只跑 50 步训练 + 20 步验证，看三件事：

- **速度**：进度条里的 `it/s`，用来估算正式训练要多久
- **显存**：另开一个终端跑 `nvidia-smi`，看 `Memory-Usage`
- **loss 在降**：真实 EEG 有规律，`train_loss` 应该比白噪声冒烟测试时明显下降（这是 QAP 在真实数据上"能学"的第一个信号）

带注释的版本：

```bash
cd /root/autodl-tmp/BioFoundation                   # 项目根目录
export DATA_PATH=/root/autodl-tmp/data              # TUAR 在 $DATA_PATH/TUAR_data/ 下
export CHECKPOINT_DIR=/root/autodl-tmp/experiments  # 输出根目录

ARGS=(
  +experiment=FEMBA_qap_tuar                        # 本次的 QAP 预训练配置
  tag=FEMBA_qap_tuar_trial                          # 试跑单独一个输出目录，不跟正式训练混在一起
  trainer.max_epochs=1                              # 只跑 1 个 epoch
  +trainer.limit_train_batches=50                   # 训练只跑 50 步（配置里没有这个键，所以要加 +）
  +trainer.limit_val_batches=20                     # 验证只跑 20 步
  trainer.log_every_n_steps=5                       # 试跑步数少，日志记得密一点
)

# -u：输出不缓冲；tee：屏幕显示的同时存一份日志
python -u run_train.py "${ARGS[@]}" 2>&1 | tee /root/autodl-tmp/qap_tuar_trial.log
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && export DATA_PATH=/root/autodl-tmp/data CHECKPOINT_DIR=/root/autodl-tmp/experiments && python -u run_train.py +experiment=FEMBA_qap_tuar tag=FEMBA_qap_tuar_trial trainer.max_epochs=1 +trainer.limit_train_batches=50 +trainer.limit_val_batches=20 trainer.log_every_n_steps=5 2>&1 | tee /root/autodl-tmp/qap_tuar_trial.log
```

另开一个终端看显存（每 5 秒刷新一次，`Ctrl+C` 退出）：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5
```

**期望**：

- 打印 `===> Loading datasets` 之后是 `49950 5550` 左右（train.h5 约 5.5 万个样本按 9:1 划分）
- 训练 50 步、验证 20 步，最后打印 `Best checkpoint path` 和验证表格，没有 Traceback
- 记下：`it/s`、显存占用、第 1 步和第 50 步附近的 `train_loss`

**用 it/s 估算正式训练时间**：每个 epoch 约 `49950 / 32 ≈ 1561` 步，10 个 epoch 约 15610 步；
时间 ≈ `15610 / it/s` 秒（再加每个 epoch 验证约 174 步）。例如 1 it/s ≈ 4.5 小时。

**结果（2026-09-27，RTX 4090）**：✅ 跑通，没有 Traceback

```
Epoch 0: 100%|█| 50/50 [00:51<00:00,  0.97it/s, v_num=7684, train_loss_step=0.201, val_loss_step=0.571, val_loss_epoch=0...
Best checkpoint path: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar_trial/FEMBA_qap_tuar_trial_27_09_16-13-43.877684/epoch=0-step=50.ckpt
Best model score: 0.5332700610160828
Validation DataLoader 0: 100%|██████████| 20/20 [00:03<00:00,  5.61it/s]
│      val_loss_epoch       │    0.5310384035110474     │
```

| 项 | 结果 |
|---|---|
| 训练速度 | **0.98 it/s**（batch 32） |
| 验证速度 | 5.61 it/s |
| 正式训练估算 | 训练 15610 步 ≈ 4.4 小时 + 验证 10 × 174 步 ≈ 5 分钟 → **约 4.5 小时** |
| 第 50 步 `train_loss_step` | 0.201（只是单个 batch 的值） |
| `val_loss_epoch` | 0.531 |
| 显存 | 这次没测到，正式训练时用 `nvidia-smi` 看 |

**现象：验证 loss（0.53）比训练 loss（0.20）高很多。** 0.20 只是第 50 步单个 batch 的值，波动大，但差距仍然明显。

**推测（待确认）**：Brevitas 的激活量化 scale 在训练开始的一段步数里是逐 batch 实时统计的（统计阶段），
验证（eval 模式）用的是另一份还没稳定的值，两边的量化 scale 不一样。才跑 50 步，很可能还在统计阶段。
如果是这样，正式训练跑过几百步后 val loss 应该明显下降、接近 train loss —— 看正式训练第 1 个 epoch 结束时的 `val_loss_epoch` 就能验证。

**从日志里补看的信息**（训练集/验证集样本数、train loss 的变化）：

```bash
grep -a -E "^[0-9]+ [0-9]+$" /root/autodl-tmp/qap_tuar_trial.log                       # 训练集/验证集样本数，期望 49973 5553 左右
grep -a -o "train_loss_step=[0-9.]*" /root/autodl-tmp/qap_tuar_trial.log | uniq         # train loss 的变化（进度条每次刷新记一个值）
```

**补看结果（2026-09-27）**：✅

```
49971 5553
train_loss_step=0.404
train_loss_step=0.283
train_loss_step=0.193
train_loss_step=0.288
train_loss_step=0.201
```

- 训练集 / 验证集：49971 / 5553（train.h5 约 5.55 万段按 9:1 划分），符合预期
- train loss 在 50 步内从 0.404 降到 0.20 左右（中间有单 batch 波动），**有下降趋势** → INT8 编码器 + Decoder 在真实 EEG 上能学到东西，可以开始正式训练
  （注意：不同数据集的 loss 绝对值不能直接比，白噪声冒烟测试的 0.15 左右跟这里没有可比性；这里只看同一次训练里的趋势）

---

## 3. 正式训练（服务器，后台运行，几个小时）

要跑几个小时，**必须放到后台**：直接在终端前台跑，SSH 一断训练就停了。这里用 `nohup ... &`，
关掉终端、断开 SSH 都不影响；日志写到文件里，随时用 `tail` 看。

带注释的版本：

```bash
cd /root/autodl-tmp/BioFoundation
export DATA_PATH=/root/autodl-tmp/data
export CHECKPOINT_DIR=/root/autodl-tmp/experiments

# nohup：忽略终端断开的信号；末尾 &：放到后台；> 日志 2>&1：标准输出和错误都写进日志文件
nohup python -u run_train.py +experiment=FEMBA_qap_tuar > /root/autodl-tmp/qap_tuar.log 2>&1 &
echo $! > /root/autodl-tmp/qap_tuar.pid             # 记下进程号，以后要停止训练用：kill $(cat /root/autodl-tmp/qap_tuar.pid)

tail -f /root/autodl-tmp/qap_tuar.log               # 实时看日志；Ctrl+C 只退出查看，不会停止训练
```

短命令版（**一条一条粘贴执行**，每条都很短，不会被终端截断）：

```bash
cd /root/autodl-tmp/BioFoundation
export DATA_PATH=/root/autodl-tmp/data CHECKPOINT_DIR=/root/autodl-tmp/experiments
nohup python -u run_train.py +experiment=FEMBA_qap_tuar > /root/autodl-tmp/qap_tuar.log 2>&1 &
echo $! > /root/autodl-tmp/qap_tuar.pid
tail -f /root/autodl-tmp/qap_tuar.log
```

正常时第 3 条只打印 `[1] 进程号`，**不会**出现 `nohup: ignoring input and appending output to 'nohup.out'`。

> **第一次启动失败的记录（2026-09-27）**：原来的单行版在粘贴时被从 `run_train.py` 后面断成了两行：
> 第 1 行 `nohup python -u run_train.py` 没带参数、用默认配置跑了（输出写进 `nohup.out`）；
> 第 2 行 `+experiment=FEMBA_qap_tuar ...` 被当成命令，报 `command not found`（`Exit 127`）；`tail -f` 在等一个空日志。
> **训练没有启动。** 处理：`Ctrl+C` 退出 tail → `ps aux | grep "[r]un_train"` 确认没有残留进程 → `rm -f nohup.out` → 用上面的短命令重新启动。
> 所以这里改成了一条一条的短命令。
>
> `nohup.out` 的最后几行证实了用默认配置跑的那次立刻就失败了（默认配置去读 TUEG，路径还是 `#CHANGEME`），没有产生任何影响：
>
> ```
> full_key: data_module.datasets.TUEG_22_channels_0
> Exception ignored in: <function HDF5Loader.__del__ ...>
> AttributeError: 'HDF5Loader' object has no attribute 'data'
> ```
>
> 后面那个 `AttributeError` 是原有的小问题：`HDF5Loader` 打开文件失败时还没设置 `self.data`，析构函数 `__del__` 又去关它，于是连带报错。
> 只在"文件打不开"时出现，不影响正常训练，暂不处理。

### 训练过程中怎么看进度

```bash
ps -p $(cat /root/autodl-tmp/qap_tuar.pid) > /dev/null && echo "running" || echo "stopped"   # 训练还在不在跑
grep -a -o "Epoch [0-9]*: *[0-9]*%" /root/autodl-tmp/qap_tuar.log | tail -1                    # 当前第几个 epoch、进度百分比
grep -a -o "val_loss_epoch=[0-9.]*" /root/autodl-tmp/qap_tuar.log | uniq                        # 每个 epoch 的验证 loss
tail -c 2000 /root/autodl-tmp/qap_tuar.log                                                      # 看日志最后一段（有报错会在这里）
```

（进度条会不停刷新，日志文件里一行很长，所以用 `grep -a -o` 只抠出需要的部分。）

### 训练结束后

```bash
ls -lh $CHECKPOINT_DIR/checkpoints/FEMBA_qap_tuar/*/     # 最优 checkpoint（epoch=...-step=....ckpt）和 last.ckpt
tail -c 3000 /root/autodl-tmp/qap_tuar.log               # 最后的 final_validate 表格
```

**期望**：`val_loss_epoch` 随 epoch 下降并逐渐变平；没有 `nan`；最后打印 `Best checkpoint path` 和验证表格。

### 可选：在 AutoDL 的 TensorBoard 里看曲线

TensorBoard 日志在 `/root/autodl-tmp/experiments/tb_logs/FEMBA_qap_tuar/`。AutoDL 的 AutoPanel 里的 TensorBoard 默认读 `/root/tf-logs`，建个软链接指过去：

```bash
rm -rf /root/tf-logs && ln -s /root/autodl-tmp/experiments/tb_logs /root/tf-logs
```

然后在控制台实例列表里打开 AutoPanel → TensorBoard（入口位置未在本实例上确认；找不到就用上面的 grep 看数值）。

### 训练中的观察

**2026-09-27，正式训练启动后第一次查看**（进度条）：

```
Epoch 2:   6%| | 100/1561 [00:47<11:37,  2.09it/s, v_num=9854, train_loss_step=0.0459, val_loss_step=0.0603, ...
```

| 项 | 结果 |
|---|---|
| 进度 | 已进入第 3 个 epoch（`Epoch 2`，从 0 数），前 2 个 epoch 已完成 |
| 速度 | **2.09 it/s**，比试跑的 0.98 快一倍（推测：`num_workers` 2→4，且 `train.h5` 已被系统缓存到内存） |
| 时间估算更新 | 每个 epoch 1561 步 ≈ 12.5 分钟，10 个 epoch **约 2 小时**（之前按试跑速度估的是 4.5 小时） |
| loss | `train_loss_step=0.0459`、`val_loss_step=0.0603` |

**验证了试跑时的推测** ✅：试跑时 val loss 0.53 远高于 train loss；现在 val loss 降到 0.06、跟 train loss 接近。
符合"Brevitas 激活量化 scale 在前几百步处于统计阶段，训练/验证用的 scale 不一致"的解释——过了统计阶段两者就一致了。

**每个 epoch 的 loss**：待补（用上面的 `grep -a -o "val_loss_epoch=..."` 查）

**结果（2026-09-27，RTX 4090）**：✅ 10 个 epoch 全部跑完，每个 epoch 约 12.5 分钟（2.09 it/s），总共约 2.1 小时

```
Epoch 9: 100%|█| 1561/1561 [12:28<00:00,  2.09it/s, v_num=9854, train_loss_step=0.120, val_loss_step=0.0492, ...
`Trainer.fit` stopped: `max_epochs=10` reached.
Best checkpoint path: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt
Best model score: 0.2583996057510376
===> Start validation
Validation DataLoader 0: 100%|██████████| 173/173 [00:20<00:00,  8.26it/s]
│      val_loss_epoch       │    0.25627410411834717    │
```

**QAP 预训练 checkpoint（后面量化微调用这个）**：

```
/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt
```

### 结果解读

1. **最优 checkpoint 就是最后一个 epoch**（epoch 9，`val_loss_epoch=0.258`）→ 到最后 loss 还在降，**还没收敛**，多训几个 epoch 可能还会更低。
   需要看每个 epoch 的曲线确认趋势有没有变平。
2. **`val_loss_epoch`（0.256）比单个 batch 的 `val_loss_step`（一直在 0.05～0.06）高很多** → 173 个 batch 里有少数 loss 特别大，把平均值拉高了。
   **推测（待确认）**：TUAR 是伪迹数据集，部分片段有很强的伪迹（检查数据时见过 -2166 ～ 3247 µV），很难重建，loss 特别大。
   如果是这样，这是数据的特点，不是训练的问题；但 `val_loss_epoch` 主要被这些"难样本"决定，拿它衡量整体重建能力会偏悲观。
   **补充发现（2026-09-27，读代码）**：输入归一化 `RobustQuartileNormalize`（`util/train_utils.py`）不是按样本/通道算四分位距，
   而是**固定的线性变换** `(x + 20) / 40`（配置里的 `quartile_normalization_lower_val: -20`、`upper_val: 20`）。
   正常 EEG（约 ±50 µV）归一化后大约在 -0.75 ～ 1.75，强伪迹（3000 µV）约 75，**大了 50 倍左右**，
   Smooth L1 在误差 > 1 时线性增长 → 这很可能就是少数片段拉高平均 loss 的原因（仍待逐样本分析确认）。
   `docs/model/FEMBA.md` 写的是"按通道 IQR 归一化"，跟代码不一致。
   推测（未验证）：输入量化是整个张量一个 scale，如果被这些极端值撑大，占多数的正常片段只能用到 INT8 的一小部分格点，精度变粗。
3. `final_validate` 重新验证得到 0.2563，跟训练中记录的 0.2584 基本一致（每次验证的随机掩码不同，有小差异）。
4. **预训练 loss 本身不能说明 QAP 有没有用**，要看下游量化微调跟"不预训练"基线的对比。

**每个 epoch 的 loss 曲线**：待补

```bash
grep -a -o "val_loss_epoch=[0-9.]*" /root/autodl-tmp/qap_tuar.log | uniq
grep -a -o "train_loss_epoch=[0-9.]*" /root/autodl-tmp/qap_tuar.log | uniq
```

---

## 之后

拿这个 checkpoint 做量化微调（跟 `docs/06_qap_finetune_step3_tuar_smoke.md` 一样的流程，但训练步数放开），
再跟**不加载预训练权重**的同样配置做对比。正式微调前还要先定问题 C（warmup 覆盖整个训练）。
