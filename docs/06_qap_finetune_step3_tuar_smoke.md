# 量化微调 第 3 步：TUAR 真实数据 + 量化微调冒烟训练

目标：用真实的 TUAR 数据，把 **加载 QAP 预训练权重 → INT8 量化微调 → 验证 → 测试** 整条流程跑一遍。
跟 QAP 预训练的冒烟测试一样，这一步只看**能不能跑完、数值正常、指标能算出来**，不看分类效果。

所有命令除特别说明外，都在 AutoDL 服务器上运行。

---

## 进度

| # | 问题 | 状态 |
|---|---|---|
| 1 | `FEMBA_quantized.yaml` 配置（含实验配置调度器问题 A、B） | ✅ 第 2 步 |
| 2 | 接口对不上（适配类） | ✅ 第 1 步 |
| 3 | 加载预训练权重的 `weights_only` | ✅ 第 2 步 |
| 4 | 带标签的数据 | ✅ 本步完成：直接用真实 TUAR，不造假数据 |
| C | warmup 覆盖整个训练 | 待决定（冒烟测试步数少，不受影响） |

---

## 数据：`TUAR_data.tar`（已处理好的 HDF5）

在 Mac 上解压 `test.h5` 检查的结果：

| 项 | 内容 |
|---|---|
| 文件 | `TUAR_data/train.h5`（12 G）、`val.h5`（1.4 G）、`test.h5`（1.6 G） |
| 结构 | 多个 group（`data_group_0`、`data_group_1`…），每个 group 下有 `X` 和 `y` |
| `X` | `(1000, 22, 1280)` float64，22 通道 × 1280 点，数值量级几百～几千 µV |
| `y` | `(N,)` int64，**只有 0 和 1** → TUAR 的**二分类（BC）**：无伪迹 / 有伪迹 |
| test 集 | 7211 个样本，0：3304，1：3907，基本均衡 |

二分类正好是 `FEMBA_quantized.yaml` 的默认设置（`num_classes: 2`、`classification_type: "bc"`），也是 INT8 分类头支持的类型
（TUAR 的 MC / MMC 逐通道任务 INT8 分类头不支持，见 `docs/04_qap_finetune_step1_adapter.md`）。

### 放到服务器上 ✅

```bash
cd /root/autodl-tmp
tar -xvf TUAR_data.tar -C /root/autodl-tmp/data     # tar 包里自带 TUAR_data/ 一层目录
ls -lh /root/autodl-tmp/data/TUAR_data/
```

结果（2026-09-27）：

```
-rwxrwxrwx 1 root root 1.6G Aug 31 10:53 test.h5
-rwxrwxrwx 1 root root  12G Aug 31 10:50 train.h5
-rwxrwxrwx 1 root root 1.4G Aug 31 10:51 val.h5
```

确认后可以 `rm /root/autodl-tmp/TUAR_data.tar` 释放 15 G（Mac 的 Downloads 里还有一份）。

---

## 改动：新建 `config/data_module/finetune_data_module_tuar.yaml`

复制自 `finetune_data_module.yaml`（默认指向 TUAB），只把路径改成 TUAR，原文件不动：

```yaml
data_module:
  _target_: data_module.finetune_data_module.FinetuneDataModule
  name: "eeg_tuar"
  cfg:
    num_workers: ${num_workers}
    batch_size: ${batch_size}
  train: {_target_: 'datasets.hdf5_dataset.HDF5Loader', hdf5_file: '${env:DATA_PATH}/TUAR_data/train.h5'}
  val:   {_target_: 'datasets.hdf5_dataset.HDF5Loader', hdf5_file: '${env:DATA_PATH}/TUAR_data/val.h5'}
  test:  {_target_: 'datasets.hdf5_dataset.HDF5Loader', hdf5_file: '${env:DATA_PATH}/TUAR_data/test.h5'}
```

---

## 冒烟训练的命令行覆盖

`FEMBA_quantized.yaml` 是按正式训练写的（4 卡 DDP、batch 256、5 个 epoch），冒烟测试在命令行里改小，**不改配置文件**：

| 覆盖 | 原值 | 作用 |
|---|---|---|
| `data_module=finetune_data_module_tuar` | `finetune_data_module`（TUAB） | 用 TUAR 数据 |
| `tag=FEMBA_qap_finetune_smoke` | `FEMBA_quantized` | 输出目录跟正式训练分开 |
| `gpus=1`、`trainer.strategy=auto` | 4 卡、`ddp` | 单卡 |
| `batch_size=32`、`num_workers=2` | 256、16 | 跟 QAP 预训练冒烟测试一致 |
| `trainer.max_epochs=1`、`+trainer.limit_train_batches=20`、`+trainer.limit_val_batches=5` | 5 个 epoch、全量 | 训练 20 步、验证 5 步 |
| `io.base_output_path=$CHECKPOINT_DIR/tb_logs` | 没设（默认 `#CHANGEME`） | TensorBoard 日志位置 |
| `pretrained_checkpoint_path='...'` | `"CHANGEME"` | 加载 QAP 预训练 checkpoint。路径里有 `=`，**必须用单引号包起来**，外面再套双引号让 shell 展开变量 |

**本地已验证（Mac，Hydra 组合 + 解析）**：✅ 所有覆盖生效

```
tag: FEMBA_qap_finetune_smoke | batch_size: 32 | num_workers: 2
model: models.FEMBA_int8.FEMBATinyInt8Classifier 2 bc
data: {'train': '/root/autodl-tmp/data/TUAR_data/train.h5', 'val': '.../val.h5', 'test': '.../test.h5'}
trainer: {'accelerator': 'gpu', 'devices': 1, 'strategy': 'auto', 'max_epochs': 1, 'limit_train_batches': 20, 'limit_val_batches': 5}
pretrained_checkpoint_path: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_27_09_05-31-02.549489/epoch=0-step=20.ckpt
final_validate/test: True True | optimizer: {'optim': 'AdamW', 'lr': 5e-05, ...} | layerwise: 0.75
```

**注意**：`final_test` 会新建一个单卡 Trainer 在**完整 test 集**上测（7211 个样本，batch 32 约 226 步），不受 `limit_*` 限制。

---

## 1. 同步到服务器（在 **Mac** 终端运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 37878" \
  config/data_module/finetune_data_module_tuar.yaml \
  docs/06_qap_finetune_step3_tuar_smoke.md \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/BioFoundation/
```

---

## 2. 跑量化微调冒烟训练（服务器）

带注释的版本（用 bash 数组写参数，每个参数一行、可以加注释；反斜杠续行的写法没法加注释）：

```bash
cd /root/autodl-tmp/BioFoundation                   # 项目根目录
export DATA_PATH=/root/autodl-tmp/data              # 数据根目录：TUAR 在 $DATA_PATH/TUAR_data/ 下
export CHECKPOINT_DIR=/root/autodl-tmp/experiments  # 输出根目录：checkpoint、TensorBoard 日志都在这下面

# 自动找到 QAP 冒烟训练最新的 checkpoint（ls -t 按修改时间从新到旧排序，head -1 取第一个）
CKPT=$(ls -t $CHECKPOINT_DIR/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt | head -1)
echo "CKPT=$CKPT"                                   # 确认找到了，应该以 epoch=0-step=20.ckpt 结尾

ARGS=(
  +experiment=FEMBA_quantized                       # 量化微调实验配置：INT8 分类器、二分类、cosine 调度器
  data_module=finetune_data_module_tuar             # 数据换成 TUAR（原配置指向 TUAB）
  tag=FEMBA_qap_finetune_smoke                      # 输出目录名，跟正式训练分开
  gpus=1                                            # 单卡（原配置 4 卡）
  trainer.strategy=auto                             # 单卡不用 DDP（原配置 ddp）
  batch_size=32                                     # 原配置 256
  num_workers=2                                     # 原配置 16
  trainer.max_epochs=1                              # 只跑 1 个 epoch（原配置 5）
  +trainer.limit_train_batches=20                   # 训练只跑 20 步；原配置里没有这个键，所以要加 +
  +trainer.limit_val_batches=5                      # 验证只跑 5 步
  io.base_output_path=$CHECKPOINT_DIR/tb_logs       # TensorBoard 日志位置（原配置没设，默认是 #CHANGEME）
  "pretrained_checkpoint_path='$CKPT'"              # 加载 QAP 预训练权重；路径里有 =，Hydra 要求用单引号包起来，
                                                    # 外面的双引号让 shell 先把 $CKPT 展开
)

# -u：输出不缓冲，实时显示；tee：屏幕显示的同时存一份完整日志
python -u run_train.py "${ARGS[@]}" 2>&1 | tee /root/autodl-tmp/qap_finetune_smoke.log
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && export DATA_PATH=/root/autodl-tmp/data CHECKPOINT_DIR=/root/autodl-tmp/experiments && CKPT=$(ls -t $CHECKPOINT_DIR/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt | head -1) && echo "CKPT=$CKPT" && python -u run_train.py +experiment=FEMBA_quantized data_module=finetune_data_module_tuar tag=FEMBA_qap_finetune_smoke gpus=1 trainer.strategy=auto batch_size=32 num_workers=2 trainer.max_epochs=1 +trainer.limit_train_batches=20 +trainer.limit_val_batches=5 io.base_output_path=$CHECKPOINT_DIR/tb_logs "pretrained_checkpoint_path='$CKPT'" 2>&1 | tee /root/autodl-tmp/qap_finetune_smoke.log
```

### 正常运行时会依次看到

| 顺序 | 输出 | 说明 |
|---|---|---|
| 1 | `CKPT=/root/.../epoch=0-step=20.ckpt` | 找到了 QAP checkpoint |
| 2 | 配置 YAML | `model._target_` 应为 `FEMBATinyInt8Classifier` |
| 3 | `===> Loading pretrained checkpoint from ...`、`Loading pretrained checkpoint`、`Pretrained model ready.` | 加载 QAP 权重 |
| 4 | 模型结构表 | |
| 5 | `Epoch 0: ... 20/20 ... train_loss...` | 训练 20 步 |
| 6 | `Best checkpoint path: .../FEMBA_qap_finetune_smoke/...` | 保存了最优 checkpoint |
| 7 | 验证指标表（`val_loss`、`val_acc` 等） | `final_validate` |
| 8 | `Re-instantiating LightningDataModule for evaluation...` | 开始 `final_test` |
| 9 | `Testing ... 226/226`，测试指标表（`test_acc`、`test_auroc` 等） | 完整 test 集 |

### 判断通过的标准

- 从头跑到测试指标表打印出来，**没有 Traceback**
- loss 是正常的有限数字（不是 `nan` / `inf`）
- 各项指标（accuracy、AUROC 等）能算出来

### 关于指标数值

现在的 QAP checkpoint 是用**白噪声假数据**预训练的，编码器没学到任何 EEG 特征；微调又只跑 20 步。
所以准确率大概在 0.5 附近（test 集里 1 类占 54%，全猜 1 也有 0.54）。**这次的数值不能用来判断 QAP 有没有效果**，
要评估 QAP，需要先用真实 EEG 数据做 QAP 预训练。

**结果（2026-09-27，第一次运行）**：❌ 创建 Trainer 时报错，**跟代码无关**

```
===> Loading pretrained checkpoint from /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_27_09_05-31-02.549489/epoch=0-step=20.ckpt
Loading pretrained checkpoint
Pretrained model ready.
Checkpoint path: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_finetune_smoke/FEMBA_qap_finetune_smoke_27_09_15-38-01.515947
...
===> Instantiate trainer
lightning_fabric.utilities.exceptions.MisconfigurationException: No supported gpu backend found!
```

- ✅ 报错之前的部分都通过了：12 个命令行覆盖全部被接受，数据模块创建成功，**QAP 预训练权重在真实训练流程里加载成功**（第 2 步 #3 的修复在真实流程中生效）
- ❌ `No supported gpu backend found!`：Lightning 找不到 GPU。推测是为了上传/解压数据切到了 AutoDL **无卡模式**

**检查命令**：

```bash
nvidia-smi --query-gpu=name --format=csv,noheader; python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

- `Permission denied` / `cuda: False` → 无卡模式：控制台关机后**正常开机（带 GPU）**，重新运行上面的单行命令（`/root/autodl-tmp` 里的数据和代码不受影响）
- `NVIDIA GeForce RTX 4090` / `cuda: True` → 是别的原因，需要进一步排查

**检查结果**：用户直接切到有卡模式后重新运行（见下）。

---

**结果（2026-09-27，第二次运行，有卡模式，新主机 `autodl-container-86b443a353`）**：❌ 启动时 `import pytorch_lightning` 就报错，**环境问题，跟代码无关**

```
File ".../scipy/special/_multiufuncs.py", line 41, in __init__
ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, ...)
```

跟之前两次一样：`numpy/_core/` 里又出现了 numpy 2.x 的 `.so` 残留文件。**这是第三次，每次都发生在换主机之后**
（5090 那台、克隆出来的 4090、这次开机换到的新主机）。

**推测（待确认）**：这些 `.so` 是 AutoDL 镜像自带的；在系统盘上 `rm` 镜像里的文件只是记了一个"删除"，
实例迁移/克隆到新主机时这个删除记录可能没带过去，镜像原文件又出现了。如果是这样，**以后每次换主机都要重新修一次**。

### 换主机后的环境修复（每次换主机后先跑）

```bash
cd /root/autodl-tmp/BioFoundation
SP=/root/miniconda3/lib/python3.12/site-packages

ls $SP/numpy/_core/ | grep "\.so" | head -3      # 有输出 = numpy 残留又出现了
pip list 2>/dev/null | grep -i nccl              # 应该只有 nvidia-nccl-cu12；出现 cu11 先停下
cat /root/constraints.txt                        # 确认约束文件还在；不在就用下面的 printf 重建

pip uninstall -y scipy numpy
rm -rf $SP/numpy $SP/numpy.libs $SP/scipy $SP/scipy.libs
pip install --no-cache-dir numpy==1.26.4 scipy==1.15.3 -c /root/constraints.txt

python -c "import torch, scipy.special, pytorch_lightning; print('ok', torch.__version__, 'nccl', torch.cuda.nccl.version(), '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

约束文件不在时：`printf "torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\nnumpy==1.26.4\n" > /root/constraints.txt`

期望最后一行：`ok 2.7.1+cu126 nccl (2, 26, 2) | cuda True NVIDIA GeForce RTX 4090`（同时确认了是有卡模式）。

**修复结果**：✅ 修复后重新运行冒烟训练，一次跑通（见下）。

---

## 最终结果（2026-09-27，RTX 4090）✅ 量化微调全流程跑通

加载 QAP 预训练权重 → INT8 量化微调 20 步 → 验证 → 加载最优 checkpoint 在**完整 TUAR test 集**（226 步，26 秒，8.45 it/s）上测试，没有 Traceback：

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃         Test metric         ┃        DataLoader 0         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test_BinaryAUROC       │     0.6057853102684021      │
│     test_BinaryAccuracy     │     0.5415337681770325      │
│ test_BinaryAveragePrecision │     0.6458893418312073      │
│    test_BinaryCohenKappa    │   -0.0005040168762207031    │
│     test_BinaryF1Score      │     0.7025373578071594      │
│    test_BinaryPrecision     │     0.5416955947875977      │
│    test_MulticlassRecall    │     0.4997674226760864      │
│          test_loss          │     0.6891332268714905      │
└─────────────────────────────┴─────────────────────────────┘
```

### 指标解读：模型基本是"全部预测为 1"，符合预期

- Accuracy 0.5415 ≈ test 集里 1 类的比例（3907 / 7211 = 0.5418）
- Precision 0.5417 也等于这个比例；F1 0.7025 = 2×0.5417/(1+0.5417)，正好是"全猜 1"时的 F1
- Cohen's Kappa ≈ 0、宏平均 Recall ≈ 0.5 → 没有超过随机水平的判别
- AUROC 0.606 略高于 0.5：输出的概率排序里有一点点信息，但 20 步训练 + 白噪声预训练，**不能作为任何结论**

原因跟预期一致：QAP checkpoint 是白噪声预训练的，微调也只跑了 20 步。**这次只证明流程能跑通。**

### 可以忽略的提示

- `No positive/negative samples in targets ...`（torchmetrics）：test 集没打乱，某些 batch 里全是同一类，逐 batch 计算时会警告，不影响最终汇总的指标
- `You called self.log('test_loss', ...) but have no logger configured`：`run_train.py` 的 `_run_test` 在没开 wandb 时用 `logger=[]`，
  所以 test 指标只打印在屏幕上（和 `tee` 的日志文件里），不写进 TensorBoard。原有行为，暂不处理
- `Be aware that when using ckpt_path, callbacks ... need to be provided`：Lightning 的通用提示，只加载模型权重做测试时无影响

---

## 本步结论 ✅

**QAP 完整流程打通：QAP 预训练 → 保存 checkpoint → 加载到 INT8 分类器 → 在真实 TUAR 数据上量化微调 → 验证 → 测试。**

未决事项：

| 事项 | 说明 |
|---|---|
| 问题 C | `warmup_epochs` = `max_epochs`，正式量化微调前要定 |
| `freeze_layers` 不生效 | 原有 bug，见 `docs/05_qap_finetune_step2_config_weights.md` |
| 深度卷积权重没伪量化 | 原版 `FEMBATinyInt8` 的限制，见 `docs/01_qap_verification.md` |
| 换主机后 numpy 残留 | 每次换主机先跑上面的"换主机后的环境修复" |
| 真正评估 QAP | 需要用真实 EEG 做 QAP 预训练，再跟对照组比较 |
