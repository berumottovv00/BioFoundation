# 第 4 步：修复验证阶段画图下标越界

所有命令都在 AutoDL 服务器上运行。每一步都给了两种写法：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，`EOF` 会失效，Python 也会报缩进错误。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 问题

`MaskTask.validation_step`（`tasks/pretrain_task.py`）在第一个验证 batch 上，会挑几个样本把"原始信号 vs 重建信号"画成图写进 TensorBoard。
挑哪几个样本是写死的：

```python
random_indices = [6, 16, 30]
```

原作者用 `batch_size: 256`，所以从来没出过问题。但我们冒烟测试用小 batch（比如 8），
取第 16、30 个样本时就会报 `IndexError: index 16 is out of bounds`：训练能正常跑完，一进验证就崩。

## 改动

`tasks/pretrain_task.py` 的 `validation_step` 里只改了一行（加一行注释）：

```python
# Fixed indices for logging signals
# 只保留小于 batch 大小的下标：batch_size < 31 时，写死的 16、30 会越界报 IndexError
random_indices = [i for i in (6, 16, 30) if i < X.shape[0]]
```

| batch 大小 | 改之前 | 改之后画图的样本 |
|---|---|---|
| ≤ 6 | IndexError | 不画图（列表为空，画图函数的循环直接跳过） |
| 7 ~ 16 | IndexError | 第 6 个 |
| 17 ~ 30 | IndexError | 第 6、16 个 |
| ≥ 31 | 正常 | 第 6、16、30 个（跟原来完全一样） |

`batch_size ≥ 31` 时行为跟原来完全一样，所以不影响原作者的大 batch 训练。

---

## 1. 把改动同步到服务器（在 Mac 上运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 端口号" tasks/pretrain_task.py root@主机:/root/autodl-tmp/BioFoundation/
```

---

## 2. 验证：用不同 batch 大小调用真实的 validation_step

不启动完整训练，直接创建 `MaskTask`（里面用的就是 QAP 模型 `FEMBATinyInt8Pretrain`），
分别用 batch = 2、8、32 调用 `validation_step`，检查不报错、画图数量对。

这一步同时顺带验证了 **MaskTask + QAP 模型 + 损失函数 + 输入归一化** 能正确地连在一起，
为第 6 步的完整训练做准备。

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "batch=|Error"
# ↑ 只显示包含 "batch=" 或 "Error" 的行，过滤掉模型配置打印和各种警告

import torch, types
from omegaconf import OmegaConf
from tasks.pretrain_task import MaskTask

# MaskTask.__init__ 只需要这四项配置（跟 Hydra 实验配置里的写法一样）
cfg = OmegaConf.create({
    'model': {'_target_': 'models.FEMBA_int8.FEMBATinyInt8Pretrain'},              # QAP 预训练模型
    'criterion': {'_target_': 'criterion.pretrain_criterion.PretrainCriterion',
                  'loss_type': 'smooth_l1'},                                       # 只在被遮住的位置算 Smooth L1
    'masking': {'patch_size': [2, 16], 'masking_ratio': 0.6},                      # 按 2×16 的块随机遮住 60%
    'input_normalization': {'normalize': True,
                            'quartile_normalization_lower_val': -20,
                            'quartile_normalization_upper_val': 20},
})
task = MaskTask(cfg).cuda()

# 没有 Trainer 时 self.logger 是 None，画图会报错；
# 换成一个假 logger：add_figure 被调用时只记录一下，不真的写 TensorBoard
figs = []
fake_logger = types.SimpleNamespace(
    experiment=types.SimpleNamespace(add_figure=lambda tag, fig, step: figs.append(tag)))
MaskTask.logger = property(lambda self: fake_logger)

torch.set_grad_enabled(False)          # 验证阶段不需要梯度，省显存

def run(B):
    figs.clear()
    X = torch.randn(B, 22, 1280, device='cuda')    # B 个假样本
    loss = task.validation_step(X, 0)              # batch_idx=0：第一个验证 batch，会触发画图
    return loss.item(), len(figs)                  # 返回 (损失值, 画了几张图)

for B in (2, 8, 32):
    loss, n = run(B)
    print(f'batch={B:2d}  loss={loss:.4f}  figures={n}')
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch, types; from omegaconf import OmegaConf; from tasks.pretrain_task import MaskTask; cfg=OmegaConf.create({'model': {'_target_': 'models.FEMBA_int8.FEMBATinyInt8Pretrain'}, 'criterion': {'_target_': 'criterion.pretrain_criterion.PretrainCriterion', 'loss_type': 'smooth_l1'}, 'masking': {'patch_size': [2, 16], 'masking_ratio': 0.6}, 'input_normalization': {'normalize': True, 'quartile_normalization_lower_val': -20, 'quartile_normalization_upper_val': 20}}); task=MaskTask(cfg).cuda(); figs=[]; fake=types.SimpleNamespace(experiment=types.SimpleNamespace(add_figure=lambda tag, fig, step: figs.append(tag))); MaskTask.logger=property(lambda self: fake); torch.set_grad_enabled(False); run=lambda B: (figs.clear(), task.validation_step(torch.randn(B,22,1280,device='cuda'), 0).item(), len(figs)); [print(f'batch={B:2d}  loss={r[1]:.4f}  figures={r[2]}') for B in (2, 8, 32) for r in [run(B)]]" 2>&1 | grep -E "batch=|Error"
```

**预期结果**：

```
batch= 2  loss=...  figures=0
batch= 8  loss=...  figures=1
batch=32  loss=...  figures=3
```

- 三行都打印出来、没有 `IndexError` → 修复生效
- `figures` 分别是 0、1、3 → 跟上面表格一致
- `loss` 是一个正常的有限数字（不是 `nan` / `inf`）→ MaskTask、QAP 模型、损失函数、归一化能正确连在一起。
  输入是随机数、模型是随机初始化，所以具体数值没有意义

可能出现、可以忽略的输出：`You are trying to self.log() but the self.trainer reference is not registered` 这类警告会被 `grep` 过滤掉；
它的意思是没有 Trainer 时 `self.log` 不记录，这是正常的。

**结果（2026-09-27，RTX 4090）**：✅

```
batch= 2  loss=0.1987  figures=0
batch= 8  loss=0.1996  figures=1
batch=32  loss=0.1988  figures=3
```

没有 `IndexError`，画图数量符合预期，loss 是正常的有限值。

---

## 附：这一步遇到的环境问题

第一次运行报错 `ValueError: All ufuncs must have type numpy.ufunc`。原因跟之前一样：`numpy/_core/` 里残留了 numpy 2.x 的 `.so` 文件。
前面的模型验证没有触发它，是因为没有导入 `pytorch_lightning`；导入 `MaskTask` 会经过 `pytorch_lightning → torchmetrics → scipy`，才暴露出来。

修复：

```bash
SP=/root/miniconda3/lib/python3.12/site-packages
pip uninstall -y scipy numpy
rm -rf $SP/numpy $SP/numpy.libs $SP/scipy $SP/scipy.libs
pip install --no-cache-dir numpy==1.26.4 scipy==1.15.3 -c /root/constraints.txt
python -c "import scipy.special, scipy.signal, pytorch_lightning; print('scipy ok')"
```

**注意**：克隆出来的新实例也可能带着这个问题，换机器后先跑一次上面最后一行检查。
