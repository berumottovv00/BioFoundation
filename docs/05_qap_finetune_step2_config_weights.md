# 量化微调 第 2 步：修模型配置（#1）+ 预训练权重加载（#3）

所有命令除特别说明外，都在 AutoDL 服务器上运行。每一步都给了两种写法：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，`EOF` 会失效，Python 也会报缩进错误。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 进度

| # | 问题 | 位置 | 状态 |
|---|---|---|---|
| 1 | `_target_` 指向不存在的文件（另外修了实验配置的调度器问题 A、B） | `config/model/FEMBA_quantized.yaml`、`config/experiment/FEMBA_quantized.yaml` | ✅ 本步完成 |
| 2 | 接口对不上 | `models/FEMBA_int8.py` | ✅ 第 1 步完成（`docs/04_qap_finetune_step1_adapter.md`） |
| 3 | 加载预训练权重的 `weights_only` 报错 | `tasks/finetune_task.py` | ✅ 本步完成 |
| 4 | 没有带标签的数据 | — | 待做 |

---

## 改动

### #1 `config/model/FEMBA_quantized.yaml`

```diff
+# FEMBA_tiny INT8 classifier for quantized fine-tuning (e.g. of QAP-pretrained
+# weights). FEMBATinyInt8Classifier adapts ARES FEMBATinyInt8 (test_25) to the
+# FinetuneTask interface, see models/FEMBA_int8.py. Supports classification_type
+# bc / mcc / ml only.
 model:
-  _target_: ARES.tests.test_networks.test_24_femba_full_expland2.FEMBATinyInt8
+  _target_: models.FEMBA_int8.FEMBATinyInt8Classifier
   seq_length: 1280
   num_channels: 22
   embed_dim: 35
   num_blocks: 2
   num_classes: ${num_classes}
   classification_type: ${classification_type}
```

只改了 `_target_`。其余参数原来就是按 FP 版 `FEMBA_finetune.yaml` 的写法（`seq_length`、`num_classes: ${num_classes}` 等），
原版 `FEMBATinyInt8` 不接受这些参数，但适配类 `FEMBATinyInt8Classifier` 正好接受，所以不用动。

### #3 `tasks/finetune_task.py` 的 `load_pretrained_checkpoint`

```diff
         print("Loading pretrained checkpoint")
-        ckpt = torch.load(model_ckpt)
+        # weights_only=False：Lightning checkpoint 里存了 Hydra 的 DictConfig，torch>=2.6 默认 weights_only=True 会拒绝加载；
+        # 这里加载的是自己训练出来的预训练 checkpoint，来源可信。
+        # map_location="cpu"：先在 CPU 上加载，之后由 Lightning 统一搬到 GPU（Brevitas 模型在 GPU 上加载会把量化内部张量重建到 CPU）
+        ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
```

- `weights_only=False`：跟 `run_train.py` 的修复同一个原因（见 `docs/03_qap_step5_6_verification.md` 问题 3）
- `map_location="cpu"`：第 1 步发现的 Brevitas 问题——模型在 GPU 上时加载权重，会把激活量化的内部张量重建到 CPU 上。
  这里强制先加载到 CPU，之后由 Lightning 统一搬到 GPU，不管 checkpoint 当初存在哪个设备上都安全

---

## 本地已做的检查（Mac）

用 Hydra 组合 `FEMBA_quantized` 实验配置并解析所有变量：

**第一次**：❌ 报错，跟 #1 无关，是实验配置自己的问题（见下一节）

```
hydra.errors.MissingConfigException: In 'defaults': Could not find 'scheduler/constant_lr'
Available options in 'scheduler': cosine, multi_step_lr
```

**命令行临时加 `scheduler=cosine` 绕过后**：✅ 模型配置正确

```
model: {'_target_': 'models.FEMBA_int8.FEMBATinyInt8Classifier', 'seq_length': 1280, 'num_channels': 22,
        'embed_dim': 35, 'num_blocks': 2, 'num_classes': 2, 'classification_type': 'bc'}
finetuning: {'freeze_layers': True} | layerwise_lr_decay: 0.75
```

参数跟适配类的构造函数完全对得上。

---

## 新发现：实验配置 `config/experiment/FEMBA_quantized.yaml` 的问题

| 问题 | 现状 | 后果 | 状态 |
|---|---|---|---|
| A | `defaults` 里 `override /scheduler: constant_lr`，但 `config/scheduler/` 下只有 `cosine`、`multi_step_lr` | 配置一组合就报 `MissingConfigException`，根本跑不起来 | ✅ 已修 |
| B | 文件末尾 `scheduler: {gamma: 0.1}` | 换成 cosine 后，`CosineLRSchedulerWrapper.__init__` 不接受 `gamma`，创建调度器时报错 | ✅ 已修 |
| C | `cosine.yaml` 默认 `warmup_epochs: 5`，本实验 `max_epochs: 5` | **整个训练都在 warmup**，学习率从 1e-6 线性升到 5e-5，从不进入 cosine 衰减。不报错，但影响训练效果 | 待决定 |

### A、B 的修改

同一个文件里写着 `scheduler_type: cosine`，FP 版 `FEMBA_finetune.yaml` 也是 `override /scheduler: cosine`，所以原意是 cosine：

```diff
 defaults:
   - override /data_module: finetune_data_module
   - override /model: FEMBA_quantized
-  - override /scheduler: constant_lr
+  - override /scheduler: cosine
   - override /task: finetune_task
   - override /criterion: finetune_criterion
 ...
-scheduler:
-  gamma: 0.1
```

### A、B 的本地验证（Mac）

1. 不加任何命令行覆盖，直接组合 `+experiment=FEMBA_quantized`：✅ 成功
   ```
   scheduler: {'_target_': 'schedulers.cosine.CosineLRSchedulerWrapper', 'trainer': {...}, 'warmup_epochs': 5, 'min_lr': 1e-06, 'warmup_lr_init': 1e-06, 't_in_epochs': False}
   scheduler_type: cosine | max_epochs: 5
   model: models.FEMBA_int8.FEMBATinyInt8Classifier
   ```
2. 调度器参数跟 `CosineLRSchedulerWrapper.__init__` 签名核对：`unexpected kwargs: [] | missing required: []` ✅
3. 按 `finetune_task.py` 的方式实际创建调度器（假设共 500 步），逐步更新学习率：✅ 能正常创建和更新
   ```
   lr at step 0/100/250/499: ['1.00e-06', '1.08e-05', '2.55e-05', '4.99e-05'] | base lr 5e-05
   ```
   —— 这一步同时暴露了问题 C：学习率全程在线性上升。

### C：warmup 覆盖了整个训练（待决定）

对比 FP 版 `FEMBA_finetune.yaml`：`warmup_epochs: 10`、`max_epochs: 30`，warmup 占 1/3。
可选做法：在 `FEMBA_quantized.yaml` 的 `scheduler:` 里显式设一个比 `max_epochs` 小的 `warmup_epochs`（例如 1），或者加大 `max_epochs`。
这是训练超参数的选择，不是 bug，**本步不改**。量化微调的冒烟测试只跑很少的步数，不受影响；正式训练前需要决定。

---

## 1. 同步到服务器（在 **Mac** 终端运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 37878" \
  config/model/FEMBA_quantized.yaml \
  config/experiment/FEMBA_quantized.yaml \
  tasks/finetune_task.py \
  docs/05_qap_finetune_step2_config_weights.md \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/BioFoundation/
```

---

## 2. 验证：用真实流程创建微调任务，加载 QAP 预训练权重

完全按 `run_train.py` 的方式：Hydra 组合 `FEMBA_quantized` 实验配置 → `hydra.utils.instantiate(cfg.task, cfg)` 创建 `FinetuneTask`
（#1：模型配置能不能正确创建出适配类）→ 调用 `task.load_pretrained_checkpoint(path)`（#3：能不能加载 QAP checkpoint）。

然后检查：

- 创建出来的模型类是 `FEMBATinyInt8Classifier`
- **编码器权重逐个核对**：checkpoint 里除 Decoder 外的每一个张量，都要跟加载后任务里的对应张量完全相等。
  `load_state_dict(strict=False)` 对名字对不上的参数会静默跳过，所以这里不看"有没有报错"，而是逐个数"真正复制过来了几个"
- 加载后能在 GPU 上走 `FinetuneTask._step` 输出分类结果

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "^(ckpt|model class|Loading|Pretrained|encoder|logits)|Error"
import glob, os, torch, hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

# run_train.py 在 import 时注册了这两个解析器，这里手动注册
OmegaConf.register_new_resolver("env", lambda k: os.getenv(k))
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

# QAP 冒烟训练生成的最新 checkpoint
path = max(glob.glob("/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt"), key=os.path.getmtime)
print("ckpt:", path)

# 跟 run_train.py 一样组合配置（问题 A、B 已修，不需要任何额外覆盖）
initialize(config_path="config", version_base="1.1")
cfg = compose(config_name="defaults", overrides=["+experiment=FEMBA_quantized"])

task = hydra.utils.instantiate(cfg.task, cfg)      # 跟 run_train.py 一样创建 FinetuneTask（#1）
print("model class:", type(task.model).__name__)   # 期望 FEMBATinyInt8Classifier

task.load_pretrained_checkpoint(path)              # #3：内部用 torch.load(..., map_location="cpu", weights_only=False)

# 逐个核对编码器张量：checkpoint 里 model.* 且不是 decoder 的，加载后必须完全相等
sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
tsd = task.state_dict()
enc = [k for k in sd if k.startswith("model.") and not k.startswith("model.decoder.")]
same = sum(torch.equal(tsd[k], sd[k]) for k in enc if k in tsd)
print("encoder tensors copied:", same, "/", len(enc))  # 期望两个数相等

# 搬到 GPU，按 FinetuneTask 的方式前向一次
task = task.cuda().eval()
x = torch.randn(4, 22, 1280, device="cuda")
out = task._step(x, task.generate_fake_mask(4, 22, 1280))
print("logits", tuple(out["logits"].shape),
      "| probs sum to 1:", bool(torch.allclose(out["probs"].sum(1), torch.ones(4, device="cuda"))),
      "| labels", out["label"].tolist())
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import glob, os, torch, hydra; from hydra import compose, initialize; from omegaconf import OmegaConf; OmegaConf.register_new_resolver('env', lambda k: os.getenv(k)); OmegaConf.register_new_resolver('get_method', hydra.utils.get_method); path=max(glob.glob('/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt'), key=os.path.getmtime); print('ckpt:', path); initialize(config_path='config', version_base='1.1'); cfg=compose(config_name='defaults', overrides=['+experiment=FEMBA_quantized']); task=hydra.utils.instantiate(cfg.task, cfg); print('model class:', type(task.model).__name__); task.load_pretrained_checkpoint(path); sd=torch.load(path, map_location='cpu', weights_only=False)['state_dict']; tsd=task.state_dict(); enc=[k for k in sd if k.startswith('model.') and not k.startswith('model.decoder.')]; same=sum(torch.equal(tsd[k], sd[k]) for k in enc if k in tsd); print('encoder tensors copied:', same, '/', len(enc)); task=task.cuda().eval(); x=torch.randn(4,22,1280,device='cuda'); out=task._step(x, task.generate_fake_mask(4,22,1280)); print('logits', tuple(out['logits'].shape), '| probs sum to 1:', bool(torch.allclose(out['probs'].sum(1), torch.ones(4,device='cuda'))), '| labels', out['label'].tolist())" 2>&1 | grep -E "^(ckpt|model class|Loading|Pretrained|encoder|logits)|Error"
```

**预期**：

```
ckpt: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_.../epoch=0-step=20.ckpt
model class: FEMBATinyInt8Classifier
Loading pretrained checkpoint
Pretrained model ready.
encoder tensors copied: N / N          （两个数相等）
logits (4, 2) | probs sum to 1: True | labels [...]
```

- `model class` 正确 → #1 修好了
- 没有 `UnpicklingError`，打印了 `Pretrained model ready.` → #3 修好了
- `encoder tensors copied` 两个数相等 → 编码器权重（含所有量化 scale）一个不漏地传进了微调任务

**结果（2026-09-27，RTX 4090）**：✅ 全部符合预期

```
ckpt: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_27_09_05-31-02.549489/epoch=0-step=20.ckpt
model class: FEMBATinyInt8Classifier
Loading pretrained checkpoint
Pretrained model ready.
encoder tensors copied: 81 / 81
logits (4, 2) | probs sum to 1: True | labels [0, 0, 0, 0]
```

- #1 ✅ 按 `run_train.py` 的方式创建出了 `FEMBATinyInt8Classifier`，实验配置不再需要任何命令行覆盖（A、B 修复生效）
- #3 ✅ QAP checkpoint 加载成功，没有 `UnpicklingError`
- 编码器 81 个张量全部逐个核对相等 —— 没有任何参数被 `strict=False` 静默跳过
- `labels` 全是 0 是正常的：分类头随机初始化、输入是随机数，还没训练，预测没有意义

---

## 本步结论 ✅

量化微调的问题 #1、#3 解决，外加修了实验配置的 A、B。**QAP 预训练权重 → 量化微调任务** 这条链路已经打通。

剩余：#4 带标签的假数据（然后跑量化微调冒烟训练）；问题 C（warmup 覆盖整个训练）正式训练前决定。

---

## 已知的其他原有问题（未修改）

- `load_pretrained_checkpoint` 里 `freeze_layers` 的逻辑：`if freeze_layers: param.requires_grad = True` —— 不管开不开都是把所有参数设为可训练，
  **实际上没有冻结任何层**。原版 FP 微调也是这样。是否需要真正冻结编码器、只训分类头，取决于实验设计，暂不处理。
