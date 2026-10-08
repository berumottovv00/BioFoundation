# QAP 对比实验：QAP 预训练 vs 从零开始，INT8 量化微调（TUAR 二分类）

目标：第一次回答"**QAP 有没有用**"。两组实验除了"是否加载 QAP 预训练权重"之外完全相同，比较 test 集上的指标。

所有命令除特别说明外，都在 AutoDL 服务器上运行。

---

## 实验设计

| 设置 | B 组：QAP（先跑） | A 组：基线（后跑） |
|---|---|---|
| 预训练权重 | **加载** `FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt`（见 `docs/07_qap_tuar_pretrain.md`） | **不加载**（随机初始化） |
| 模型 | `FEMBATinyInt8Classifier`（INT8 编码器 + INT8 分类头），二分类 `bc` | 同左 |
| 数据 | TUAR：训练 `train.h5`、验证 `val.h5`、最终测试 `test.h5` | 同左 |
| epoch | 5 | 同左 |
| 学习率 | AdamW 5e-5，weight decay 0.5，分层衰减 0.75，cosine：**warmup 1 个 epoch** 后降到 1e-6 | 同左 |
| batch | 32 | 同左 |
| 随机种子 | 42：数据顺序、分类头初始化都相同 | 同左 |
| 输出 tag | `FEMBA_qap_ft_B_qap` | `FEMBA_qap_ft_A_scratch` |

跟 `FEMBA_quantized.yaml` 原值的差别（全部在命令行覆盖，不改配置文件）：

| 覆盖 | 原值 | 原因 |
|---|---|---|
| `data_module=finetune_data_module_tuar` | TUAB | 用 TUAR |
| `gpus=1`、`trainer.strategy=auto` | 4 卡、DDP | 单卡 |
| `batch_size=32`、`num_workers=4` | 256、16 | 单卡 256 显存不一定够；32 在 QAP 预训练里跑过 |
| `scheduler.warmup_epochs=1` | 5（来自 `cosine.yaml`） | **问题 C**：原来 warmup = max_epochs = 5，整个训练都在 warmup |
| A 组 `pretrained_checkpoint_path=null` | `"CHANGEME"` | 不覆盖的话会去加载一个不存在的文件 |

**已知限制**：

- `freeze_layers` 不生效（原有 bug），两组都是**整个模型一起训练**
- QAP 预训练数据是 TUAR 自己的 `train.h5`（没用标签），规模小，结论是初步的
- 每组只跑一个随机种子，指标差距如果很小，可能只是随机波动

---

## 改动：新建 `scripts/run_qap_finetune_compare.sh`

把两组放进一个脚本按顺序跑（先 B 后 A），避免长命令粘贴被截断；每组日志单独一个文件：

- B 组：`/root/autodl-tmp/qap_ft_B_qap.log`
- A 组：`/root/autodl-tmp/qap_ft_A_scratch.log`
- 脚本本身（开始/结束时间、退出码）：`/root/autodl-tmp/qap_ft_compare.log`

脚本开头会检查 QAP checkpoint 存不存在，不存在直接退出。checkpoint 路径可以用环境变量 `QAP_CKPT` 覆盖。

---

## 本地已做的检查（Mac）

用一个假的 `python` 截获脚本实际传给 `run_train.py` 的参数，再用 Hydra 分别组合两组的完整配置并逐项对比：✅

```
bash syntax ok
first run (B) tag: FEMBA_qap_ft_B_qap | second run (A) tag: FEMBA_qap_ft_A_scratch
differences B vs A: {'tag': ('FEMBA_qap_ft_B_qap', 'FEMBA_qap_ft_A_scratch'),
                     'pretrained_checkpoint_path': ('.../epoch=9-step=15610.ckpt', None)}
shared: {'model._target_': 'models.FEMBA_int8.FEMBATinyInt8Classifier', 'model.num_classes': 2, 'model.classification_type': 'bc',
         'batch_size': 32, 'num_workers': 4, 'trainer.devices': 1, 'trainer.strategy': 'auto', 'trainer.max_epochs': 5,
         'scheduler.warmup_epochs': 1, 'optimizer.lr': 5e-05, 'layerwise_lr_decay': 0.75, 'seed': 42, 'final_test': True}
data: /root/autodl-tmp/data/TUAR_data/train.h5 /root/autodl-tmp/data/TUAR_data/test.h5
```

- 顺序：先 B 后 A ✅
- **两组只有 `tag` 和 `pretrained_checkpoint_path` 不同** ✅

按 `finetune_task.py` 的方式创建调度器，假设共 5 × 1735 步，看学习率（问题 C 修复后）：✅

```
lr at step 0 / 1735 (end of warmup) / 4000 / 8674: ['1.00e-06', '4.53e-05', '2.85e-05', '1.00e-06']
```

第 1 个 epoch 从 1e-6 升到约 5e-5，之后 cosine 降回 1e-6，不再整个训练都在 warmup。

---

## 时间估算

`train.h5` 约 5.55 万段，batch 32 → 每个 epoch 约 1735 步。按 QAP 预训练的 ~2 it/s：每个 epoch 约 15 分钟，
5 个 epoch + 每个 epoch 的验证 + 最终测试 ≈ **每组 1.3 小时，两组共约 2.5 小时**。B 组的结果大约 1.3 小时后先出来。

---

## 0. 换主机后先检查环境（服务器）

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch, scipy.special, pytorch_lightning, brevitas; print('ok', torch.__version__, 'nccl', torch.cuda.nccl.version(), '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

报 `All ufuncs must have type numpy.ufunc` → 按 `docs/06_qap_finetune_step3_tuar_smoke.md` 的"换主机后的环境修复"处理。

---

## 1. 同步到服务器（在 **Mac** 终端运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 37878" \
  scripts/run_qap_finetune_compare.sh \
  docs/08_qap_finetune_compare.md \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/BioFoundation/
```

---

## 2. 启动（服务器，**一条一条粘贴执行**）

```bash
cd /root/autodl-tmp/BioFoundation
ls -lh /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/
nohup bash scripts/run_qap_finetune_compare.sh > /root/autodl-tmp/qap_ft_compare.log 2>&1 &
echo $! > /root/autodl-tmp/qap_ft_compare.pid
sleep 5; cat /root/autodl-tmp/qap_ft_compare.log
```

- 第 2 条：确认 QAP checkpoint `epoch=9-step=15610.ckpt` 在
- 第 3 条：终端只打印 `[1] 进程号`
- 最后一条：应该看到 `B: QAP-pretrained encoder (...)`

日志文件第一行的 `nohup: ignoring input` 是正常的：nohup 的这句提示写到了 stderr，而 stderr 被重定向进了日志文件。
**不正常**的是 `nohup: ignoring input and appending output to 'nohup.out'`——那表示输出重定向没生效（`docs/07_qap_tuar_pretrain.md` 里第一次启动失败就是这样）。

**启动结果（2026-09-27）**：✅

```
total 200M
-rw-r--r-- 1 root root 100M Sep 27 18:29 'epoch=9-step=15610.ckpt'
-rw-r--r-- 1 root root 100M Sep 27 18:29  last.ckpt
[1] 6968
nohup: ignoring input
[2026-09-27 18:53:34] B: QAP-pretrained encoder (/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt)
```

QAP checkpoint 在（100M），脚本已启动（进程号 6968），B 组 18:53 开始。预计 B 组约 20:10 结束，A 组约 21:30 结束。

---

## 3. 查看进度（服务器，任何终端都可以）

```bash
cat /root/autodl-tmp/qap_ft_compare.log                                                   # 现在跑到哪一组、已结束的组的退出码
ps -p $(cat /root/autodl-tmp/qap_ft_compare.pid) > /dev/null && echo "running" || echo "stopped"

grep -a -o "Epoch [0-9]*: *[0-9]*%" /root/autodl-tmp/qap_ft_B_qap.log | tail -1           # B 组当前进度
grep -a -o "Epoch [0-9]*: *[0-9]*%" /root/autodl-tmp/qap_ft_A_scratch.log | tail -1       # A 组当前进度（B 跑完后才有）

grep -a "Loading pretrained checkpoint\|Pretrained model ready\|No pretrained checkpoint" /root/autodl-tmp/qap_ft_B_qap.log /root/autodl-tmp/qap_ft_A_scratch.log
# 期望：B 组两行 "Loading..." + "Pretrained model ready."；A 组一行 "No pretrained checkpoint provided"

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

**要停止整个实验**：`pkill -f run_qap_finetune_compare.sh; pkill -f run_train.py`

### 查看 loss

`FinetuneTask` 记录的 loss（`tasks/finetune_task.py`）：

| 名字 | 什么时候记录 | 进度条上显示为 |
|---|---|---|
| `train_loss` | 每一步 + 每个 epoch | `train_loss_step`（**当前 batch**）、`train_loss_epoch`（**上一个 epoch 的平均**） |
| `val_loss` | 每个 epoch 结束验证时 | `val_loss` |
| `test_loss` | 最后测试时 | — |

当前 epoch 的平均 loss 要等这个 epoch 跑完才算出来；跑到一半时只能看最新 batch 的值。

```bash
# 1. 进度条最新一次刷新的完整内容：epoch、进度、train_loss_step、上个 epoch 的 train_loss_epoch 和 val_loss
tr '\r' '\n' < /root/autodl-tmp/qap_ft_B_qap.log | grep -a "^Epoch" | tail -1

# 2. 当前 epoch 里 loss 的走势（进度条每刷新一次记一个值，取最近 20 个）
tr '\r' '\n' < /root/autodl-tmp/qap_ft_B_qap.log | grep -a -o "Epoch [0-9]*: *[0-9]*%.*train_loss_step=[0-9.]*" | sed 's/|.*train_loss_step=/ train_loss_step=/' | tail -20

# 3. 每个已完成 epoch 的平均 train loss 和 val loss
grep -a -o "train_loss_epoch=[0-9.]*" /root/autodl-tmp/qap_ft_B_qap.log | uniq
grep -a -o "val_loss=[0-9.]*" /root/autodl-tmp/qap_ft_B_qap.log | uniq
```

`tr '\r' '\n'`：进度条靠回车符 `\r` 在同一行反复刷新，整个日志看起来只有一行；换成换行后每次刷新变成单独一行，才能用 `tail` 取最新的。
A 组把 `qap_ft_B_qap` 换成 `qap_ft_A_scratch`。

### 查看每个 epoch 的验证指标

从 TensorBoard 日志里直接读出每个 epoch 的验证指标（AUROC、准确率等），每行一个指标，列表是各 epoch 的值：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import glob, os; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E; d=max(glob.glob('/root/autodl-tmp/experiments/tb_logs/FEMBA_qap_ft_B_qap/*'), key=os.path.getmtime); print('log dir:', d); e=E(d); e.Reload(); [print(t, [round(s.value, 4) for s in e.Scalars(t)]) for t in sorted(e.Tags()['scalars']) if t.startswith('val') and not t.endswith('_step')]"
```

A 组把 `FEMBA_qap_ft_B_qap` 换成 `FEMBA_qap_ft_A_scratch`。

**注意**：这是**验证集**（`val.h5`）上的指标，只用来看训练过程；两组最终比较用的是 `test.h5` 上的测试结果（第 4 节）。

### 训练中的观察

**2026-09-27，B 组启动后查看**：✅

```
===> Loading pretrained checkpoint from /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt
Loading pretrained checkpoint
Pretrained model ready.
Epoch 3:  65%
memory.used [MiB], memory.total [MiB]
5594 MiB, 12288 MiB
```

- B 组确实加载了 QAP 权重 ✅
- 已到第 4 个 epoch 的 65%，比估算快
- 显存占用 5.6 GB
- **待确认**：显存总量显示 12288 MiB（12 GB），RTX 4090 应为 24 GB——这次开机换了主机，GPU 型号可能不是 4090。
  不影响实验公平性（A、B 两组由同一个脚本在同一台机器上依次跑），但记录结果时要写清楚 GPU 型号。
  查看：`nvidia-smi --query-gpu=name,memory.total --format=csv`

**B 组每个 epoch 的 val loss**（2026-09-27，第 5 个 epoch 进行中时查看）：

```
val_loss=0.434
val_loss=0.416
val_loss=0.403
val_loss=0.414
```

**怎么理解这个数**：`val_loss` 是二分类交叉熵。

| 情况 | 交叉熵 |
|---|---|
| 完全随机（两类各给 50%） | ln 2 ≈ 0.693 |
| 只按类别比例猜（冒烟测试时"全猜 1"的状态） | ≈ 0.69 |
| **B 组** | **0.40～0.43** |
| 完美 | 0 |

- 从 0.69 降到 0.40 → 模型确实学到了区分能力。e^-0.41 ≈ 0.66：给正确类别的概率平均（几何平均）约 0.66
- 单看 loss 判断不了好坏，要跟 **A 组（从零开始）** 比；分类效果看 AUROC 更直观
- 参考上限：仓库文档里 FEMBA-Base 在 TUAR BC 上 **AUROC 0.949**（`docs/model/FEMBA.md`）。但那是 47.7M 参数、FP32、TUEG 大规模预训练；
  我们是 7.8M 参数 INT8 tiny、只在 TUAR train 上预训练，**不能直接对比**
- 第 3 个 epoch 最低（0.403），第 4 个回升到 0.414：可能是波动，也可能开始过拟合，单次运行判断不了。
  checkpoint 按 val_loss 保存最优的，最终 test 用的就是 val_loss 最低的那个 epoch，不受后面回升影响

**B 组训练完成**（2026-09-27 查看）：`Epoch 4: 100%`，5 个 epoch 全部跑完，显存稳定在 ~5.6 GB。
接下来脚本自动做：`final_validate`（val loss 最低的 checkpoint）→ `final_test`（完整 test 集）→ 开始 A 组。

**B 组每个 epoch 的验证指标（AUROC 等）**：待补

**B 组 test 结果 + A 组是否已开始**：

```bash
cat /root/autodl-tmp/qap_ft_compare.log; grep -a -A12 "Test metric" /root/autodl-tmp/qap_ft_B_qap.log
```

期望：`B finished, exit code 0`、`A: from scratch (random init)`，以及 B 组的 test 指标表。结果：B 组 test 表见第 4 节 ✅

**脚本日志**（2026-09-27 查看）：

```
nohup: ignoring input
[2026-09-27 18:53:34] B: QAP-pretrained encoder (.../FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt)
[2026-09-27 20:18:40] B finished, exit code 0
[2026-09-27 20:18:40] A: from scratch (random init)
```

- B 组 18:53 → 20:18，**用时 85 分钟**，正常结束（exit code 0）
- A 组 20:18 开始，按 B 组用时推算约 **21:45** 结束
- **更新**：`[2026-09-27 21:43:32] A finished, exit code 0` → A 组也正常结束，用时 85 分钟 ✅
- A 组日志里有 `No pretrained checkpoint provided. Proceeding without loading.` → 确认 A 组**没有**加载预训练权重，对比设置正确 ✅

**A 组进度**：

```bash
tr '\r' '\n' < /root/autodl-tmp/qap_ft_A_scratch.log | grep -a "^Epoch" | tail -1
ps -p $(cat /root/autodl-tmp/qap_ft_compare.pid) > /dev/null && echo "running" || echo "stopped"
```

`stopped` 但 `qap_ft_compare.log` 里没有 `A finished` → A 组中途出错，用 `tail -c 3000 /root/autodl-tmp/qap_ft_A_scratch.log` 看报错。

---

## 4. 结果：对比两组的 test 指标

两组都结束后（`qap_ft_compare.log` 里出现 `A finished, exit code 0`）：

```bash
for f in qap_ft_B_qap qap_ft_A_scratch; do echo "===== $f"; grep -a -A12 "Test metric" /root/autodl-tmp/$f.log; done
```

**结果（2026-09-27，两组都已完成）** ✅

B 组原始输出：

```
│      test_BinaryAUROC       │     0.9013848304748535      │
│     test_BinaryAccuracy     │     0.8159756064414978      │
│ test_BinaryAveragePrecision │     0.9175984859466553      │
│    test_BinaryCohenKappa    │     0.6330193281173706      │
│     test_BinaryF1Score      │     0.8203600645065308      │
│    test_BinaryPrecision     │     0.8706896305084229      │
│    test_MulticlassRecall    │     0.8196662664413452      │
│          test_loss          │     0.41963112354278564     │
```

A 组原始输出：

```
│      test_BinaryAUROC       │     0.8727251887321472      │
│     test_BinaryAccuracy     │     0.8033559918403625      │
│ test_BinaryAveragePrecision │     0.8927953243255615      │
│    test_BinaryCohenKappa    │     0.6079531311988831      │
│     test_BinaryF1Score      │     0.8077548742294312      │
│    test_BinaryPrecision     │     0.8587489128112793      │
│    test_MulticlassRecall    │     0.8070862293243408      │
│          test_loss          │     0.46616053581237793     │
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226/226 0:00:27 • 0:00:00 8.32it/s
```

| 指标 | B：QAP | A：从零开始 | 差值（B − A） | 参考：FEMBA-Base（FP32，TUEG 预训练） |
|---|---|---|---|---|
| test_BinaryAUROC | **0.9014** | 0.8727 | **+0.0287** | 0.949 |
| test_BinaryAveragePrecision（AUPR） | **0.9176** | 0.8928 | **+0.0248** | 0.932 |
| test_BinaryAccuracy | **0.8160** | 0.8034 | +0.0126 | — |
| test_BinaryF1Score | **0.8204** | 0.8078 | +0.0126 | — |
| test_BinaryCohenKappa | **0.6330** | 0.6080 | +0.0251 | — |
| test_BinaryPrecision | **0.8707** | 0.8587 | +0.0119 | — |
| test_MulticlassRecall（宏平均） | **0.8197** | 0.8071 | +0.0126 | — |
| test_loss | **0.4196** | 0.4662 | −0.0465（越低越好） | — |

参考列来自 `docs/model/FEMBA.md`：模型规模、精度、预训练数据都不同，只能当上限参考，不能直接比。

## 结论

**QAP 预训练在所有 8 项 test 指标上都优于从零开始**：AUROC +0.029、AUPR +0.025、Kappa +0.025、准确率 / F1 / Recall 各 +0.013，test loss 低 0.047。

**这个差距有多可信**：

- **test 集抽样误差**：按 Hanley–McNeil 公式近似，7211 个样本（正 3907 / 负 3304）上 AUROC 的标准误约 0.004（B 0.0036、A 0.0041）。
  AUROC 差 0.029 远大于这个量级，而且两组用的是同一个 test 集（配对比较，差值的误差更小）→ **不太可能只是 test 集抽样的偶然**
- **训练随机性（没测）**：每组只跑了 1 个随机种子。不同种子之间训练结果的波动有多大，目前不知道；如果波动接近 0.03，结论就站不住。
  **需要每组跑多个种子（例如 3 个）报告均值 ± 标准差，才能下正式结论**
- 两组除预训练权重外设置完全相同（本地逐项核对过配置），数据顺序、分类头初始化也相同（种子 42）

**初步结论**：在 TUAR 二分类上，QAP 预训练让 INT8 FEMBA-tiny 的 AUROC 从 0.873 提升到 0.901，提升一致地体现在所有指标上；
正式结论还需要多种子实验确认。另外预训练数据是 TUAR 自己的 train 划分（未用标签）、规模小，能否推广到更大的预训练数据和其他任务还要验证。
