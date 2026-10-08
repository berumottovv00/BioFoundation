# 量化微调 第 1 步：分类适配类 `FEMBATinyInt8Classifier`

所有命令除特别说明外，都在 AutoDL 服务器上运行。每一步都给了两种写法：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，`EOF` 会失效，Python 也会报缩进错误。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 背景：量化微调要解决的 4 个问题

QAP 流程分两段：**QAP 预训练**（无标签，已跑通）→ **量化微调**（有标签，训练 INT8 分类头）。量化微调现在跑不通，有 4 个问题：

| # | 问题 | 位置 | 状态 |
|---|---|---|---|
| 1 | `_target_` 指向不存在的文件（`test_24_femba_full_expland2`），参数名也对不上 | `config/model/FEMBA_quantized.yaml` | 待做 |
| 2 | **接口对不上**：`FinetuneTask` 调 `self.model(X, mask)` 期望返回 `(logits, _)`；`FEMBATinyInt8` 是 `forward(x)`、输入 4 维、只返回 logits | `tasks/finetune_task.py:168` | ✅ 本步完成 |
| 3 | 加载预训练权重用 `torch.load(ckpt)`，会碰到 `weights_only` 报错 | `tasks/finetune_task.py:115` | 待做 |
| 4 | 没有带标签的数据 | — | 待做 |

---

## 改动：`models/FEMBA_int8.py` 新增 `FEMBATinyInt8Classifier`

跟 `FEMBATinyInt8Pretrain` 一样继承 `FEMBATinyInt8`，ARES 原文件不改。**只做接口转换，分类计算完全用父类的 forward**：

```python
class FEMBATinyInt8Classifier(FEMBATinyInt8):
    def __init__(self, seq_length=1280, num_channels=22, embed_dim=35, num_blocks=2, exp=4,
                 d_state=16, d_conv=4, patch_size=(2, 16), stride=(2, 16),
                 num_classes=2, classification_type="bc"):
        # classification_type 不是 bc/mcc/ml 就报 NotImplementedError
        # patch_size/stride 转成 tuple（Hydra 传进来的是 ListConfig）
        super().__init__(inp_size=(num_channels, seq_length), ..., num_classes=num_classes)
        self.classification_type = classification_type

    def forward(self, x, mask=None):        # x: (B, C, T)；mask 忽略（FinetuneTask 传的是全 False）
        logits = super().forward(x.unsqueeze(1))   # 父类要求 [B, 1, H, W]
        return logits, x                           # 跟 FP 版 FEMBA 一样返回 (logits, 原始信号)
```

| | `FEMBATinyInt8`（ARES 原版） | `FEMBATinyInt8Classifier`（适配类） |
|---|---|---|
| 构造参数 | `inp_size=(22,1280)`、`expand`、`num_classes` | `seq_length`、`num_channels`、`exp`、`num_classes`、`classification_type`（跟 FP 版 FEMBA 配置一致） |
| `forward` | `forward(x)`，x 是 `[B,1,H,W]` | `forward(x, mask)`，x 是 `[B,C,T]` |
| 返回 | `logits` | `(logits, x)` |
| 计算 | INT8 编码器 → 池化 → INT8 线性层 | **完全相同**（直接调父类） |
| 参数名 | — | **完全相同**，所以 QAP 预训练权重能直接加载 |

**限制**：原版分类头是"全局平均池化 + 一个 INT8 线性层"，每个样本只输出一个类别，所以只支持 `bc`（二分类）、`mcc`（多分类）、`ml`；
不支持 FP 版的逐通道分类 `mc`、`mmc`，传进来会直接报 `NotImplementedError`。

`FinetuneTask` 对模型的其他要求也都满足：有 `classifier` 属性（`:113`）；分层学习率按参数名里的 `mamba_blocks.N` / `norm_layers.N` 识别层号（`:280`），父类参数名正好是这个格式。

---

## 1. 同步到服务器（在 **Mac** 终端运行）

```bash
cd /Users/shiyu/PyCharmMiscProject/BioFoundation
rsync -avR -e "ssh -p 37878" \
  models/FEMBA_int8.py \
  docs/04_qap_finetune_step1_adapter.md \
  root@connect.westc.seetacloud.com:/root/autodl-tmp/BioFoundation/
```

---

## 2. 验证 A：适配类跟原版分类模型等价

拿一个原版 `FEMBATinyInt8`，把它的权重**严格**加载到适配类里（`strict=True`，名字或形状有任何不同都会报错），
然后用同一个输入分别算，logits 必须逐位相同。

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "^(strict|logits|returns|mc )|Error"
import torch
from models.FEMBA_int8 import FEMBATinyInt8Classifier as A                      # A：适配类
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C   # C：原版分类模型

torch.manual_seed(0)
c = C(num_classes=2).cuda().eval()                 # 原版；eval() 避免 Brevitas 训练模式下更新量化统计量
a = A(num_classes=2, classification_type="bc").cuda().eval()

a.load_state_dict(c.state_dict(), strict=True)     # 严格加载：结构不同就会直接报错
print("strict load: OK")

x = torch.randn(4, 22, 1280, device="cuda")        # FinetuneTask 喂进来的形状 (B, C, T)
mask = torch.zeros_like(x, dtype=torch.bool)       # FinetuneTask 传的就是全 False 的假 mask

logits_a, x_out = a(x, mask)                       # 适配类：forward(x, mask) -> (logits, x)
logits_c = c(x.unsqueeze(1))                       # 原版：forward([B,1,H,W]) -> logits

print("logits shape", tuple(logits_a.shape), "| identical:", torch.equal(logits_a, logits_c))
print("returns input unchanged:", x_out is x)

try:                                               # 不支持的分类类型应该直接报错
    A(classification_type="mc")
    print("mc NOT rejected")
except NotImplementedError:
    print("mc rejected: OK")
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C(num_classes=2).cuda().eval(); a=A(num_classes=2, classification_type='bc').cuda().eval(); a.load_state_dict(c.state_dict(), strict=True); print('strict load: OK'); x=torch.randn(4,22,1280,device='cuda'); mask=torch.zeros_like(x,dtype=torch.bool); la,xo=a(x,mask); lc=c(x.unsqueeze(1)); print('logits shape', tuple(la.shape), '| identical:', torch.equal(la,lc)); print('returns input unchanged:', xo is x); exec('try:\n A(classification_type=\"mc\"); print(\"mc NOT rejected\")\nexcept NotImplementedError:\n print(\"mc rejected: OK\")')" 2>&1 | grep -E "^(strict|logits|returns|mc )|Error"
```

**预期**：

```
strict load: OK
logits shape (4, 2) | identical: True
returns input unchanged: True
mc rejected: OK
```

**结果（2026-09-27，第一次运行）**：❌ `strict=True` 加载失败

```
RuntimeError: Error(s) in loading state_dict for FEMBATinyInt8Classifier:
```

具体缺了/多了哪些参数被 `grep` 过滤掉了，还没看到。

**推测（待确认）**：可能是验证脚本的问题，不是适配类的问题。原版 `c` 创建后没在训练模式下跑过前向就导出了权重；
Brevitas 的激活量化 scale（`scaling_impl.value`）可能要在训练模式下收集过统计量后才会写进 `state_dict`，导致导出的字典缺这些 key。
（第 1 层验证里预训练模型在训练模式下跑过一次前向，当时没出现这个问题，跟推测吻合。）

**诊断命令**：不做加载，只比较两个模型 `state_dict` 的参数名，看各自多了哪些：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C(num_classes=2).cuda().eval(); a=A(num_classes=2, classification_type='bc').cuda().eval(); ks=set(c.state_dict()); ka=set(a.state_dict()); print('c keys', len(ks), '| a keys', len(ka)); print('only in a:', sorted(ka-ks)[:3], len(ka-ks)); print('only in c:', sorted(ks-ka)[:3], len(ks-ka))" 2>&1 | grep -E "keys|only in|Error"
```

- `only in a` 全是 `...scaling_impl.value` 这类 → 推测成立，只需改验证脚本，适配类没问题
- 出现别的名字 → 适配类结构跟原版确实不同，需要查

**诊断结果**：

```
c keys 49 | a keys 49
only in a: [] 0
only in c: [] 0
```

参数名完全相同 → **适配类结构跟原版一致，问题在验证脚本**。
线索：模型有 84 个参数，`state_dict` 却只有 49 个 key，少掉的约 35 个正是各量化层的 `scaling_impl.value`。

**结论**：新建的 Brevitas 模型在训练模式下跑过前向之前，激活量化 scale 还没初始化；
导出 `state_dict` 时 Brevitas 会跳过它们，加载时却要求它们存在 → `strict=True` 报 `Missing key`。
真实流程不受影响：QAP 预训练出来的 checkpoint 已经训练过，scale 都初始化了（验证 B 用的就是它）。

**修正**：导出前先让原版 `c` 在训练模式下跑一次前向（warmup），再切回 `eval()`。

### 修正后的验证 A

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "^(value|strict|logits|returns|mc )|Error"
import torch
from models.FEMBA_int8 import FEMBATinyInt8Classifier as A
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C

torch.manual_seed(0)
c = C(num_classes=2).cuda()
x = torch.randn(4, 22, 1280, device="cuda")
nv = lambda m: sum("scaling_impl.value" in k for k in m.state_dict())   # 数一下 state_dict 里有几个激活 scale

print("value keys before warmup:", nv(c))          # 期望 0：还没初始化，导出时被跳过
c.train(); c(x.unsqueeze(1)); c.eval()             # warmup：训练模式跑一次前向，初始化所有激活 scale
print("value keys after warmup:", nv(c))           # 期望 30 多个

a = A(num_classes=2, classification_type="bc").cuda().eval()
a.load_state_dict(c.state_dict(), strict=True)     # 严格加载：结构不同就会直接报错
print("strict load: OK")

mask = torch.zeros_like(x, dtype=torch.bool)       # FinetuneTask 传的就是全 False 的假 mask
logits_a, x_out = a(x, mask)                       # 适配类：forward(x, mask) -> (logits, x)
logits_c = c(x.unsqueeze(1))                       # 原版：forward([B,1,H,W]) -> logits
print("logits shape", tuple(logits_a.shape), "| identical:", torch.equal(logits_a, logits_c))
print("returns input unchanged:", x_out is x)

try:                                               # 不支持的分类类型应该直接报错
    A(classification_type="mc")
    print("mc NOT rejected")
except NotImplementedError:
    print("mc rejected: OK")
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C(num_classes=2).cuda(); x=torch.randn(4,22,1280,device='cuda'); nv=lambda m: sum('scaling_impl.value' in k for k in m.state_dict()); print('value keys before warmup:', nv(c)); c.train(); c(x.unsqueeze(1)); c.eval(); print('value keys after warmup:', nv(c)); a=A(num_classes=2, classification_type='bc').cuda().eval(); a.load_state_dict(c.state_dict(), strict=True); print('strict load: OK'); mask=torch.zeros_like(x,dtype=torch.bool); la,xo=a(x,mask); lc=c(x.unsqueeze(1)); print('logits shape', tuple(la.shape), '| identical:', torch.equal(la,lc)); print('returns input unchanged:', xo is x); exec('try:\n A(classification_type=\"mc\"); print(\"mc NOT rejected\")\nexcept NotImplementedError:\n print(\"mc rejected: OK\")')" 2>&1 | grep -E "^(value|strict|logits|returns|mc )|Error"
```

**预期**：

```
value keys before warmup: 0
value keys after warmup: 30 多个
strict load: OK
logits shape (4, 2) | identical: True
returns input unchanged: True
mc rejected: OK
```

**结果（2026-09-27，修正后第一次运行）**：部分通过 ⚠️

```
value keys before warmup: 0
value keys after warmup: 35
strict load: OK
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

- ✅ warmup 前 0 个、warmup 后 35 个激活 scale → 上面"未初始化的 scale 导出时被跳过"的结论成立
- ✅ `strict load: OK` → 35 个 scale 加上其余参数全部严格加载成功，**适配类结构跟原版完全一致**
- ❌ 加载后前向时报设备不一致：有张量留在了 CPU 上

**推测（待确认）**：Brevitas 加载 scale 时可能创建了一个没注册成参数/buffer 的普通张量，放在 CPU 上，不会跟着 `.cuda()` 走。
如果是这样，真实微调流程（Lightning 先加载权重、再把模型搬到 GPU）也会碰到，必须查清楚。
也可能是原版 `c` 在"warmup 后切 eval"这个状态下的问题，跟加载无关。

**诊断命令**：把前向拆成两次，用 `>>>` 标记区分报错发生在加载过权重的适配类 `a` 上，还是 warmup 后的原版 `c` 上；保留最后 30 行调用栈：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C(num_classes=2).cuda(); x=torch.randn(4,22,1280,device='cuda'); c.train(); c(x.unsqueeze(1)); c.eval(); a=A(num_classes=2, classification_type='bc').cuda().eval(); a.load_state_dict(c.state_dict(), strict=True); print('>>> a forward'); a(x, None); print('>>> a OK'); print('>>> c forward'); c(x.unsqueeze(1)); print('>>> c OK')" 2>&1 | grep -v -E "Warning|warn|rename|^\s*$|Configuration|Input:|Grid:|d_model|d_inner|d_state|bit width|Weight size|Total for" | tail -30
```

**诊断结果（2026-09-27）**：报错发生在**加载过权重的适配类 `a`** 上（打印了 `>>> a forward`，没有 `>>> a OK`）。调用栈末尾：

```
brevitas/nn/quant_linear.py            forward → inner_forward_impl → linear(x, quant_weight, quant_bias)
brevitas/quant_tensor/int_torch_handler.py  quant_output_scale_impl:
    output_scale = quant_weight_scale * quant_input_scale
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

某个 `QuantLinear` 算输出 scale 时，权重 scale 和输入 scale 一个在 GPU、一个在 CPU。
加载进来的值来自 `c`，本身都在 GPU 上，所以留在 CPU 的应该是 `a` 自己的某个内部张量。具体是哪个还没定位（调用栈顶部被 `tail` 截掉了）。

### 诊断 2：找出留在 CPU 上的张量 + 模拟真实微调的加载顺序

遍历每个子模块，列出所有在 CPU 上的张量（参数、buffer、以及没注册的普通张量属性），分三种情况：

1. `a` **加载前**（已 `.cuda()`）：有没有本来就留在 CPU 的
2. `a` **加载后**（先 `.cuda()` 再加载）：是不是加载时在 CPU 上新建了张量
3. `a2`：**先在 CPU 上加载、再 `.cuda()`** —— Lightning 真实微调的顺序。如果这样前向没问题，说明只是验证脚本"先 `.cuda()` 再加载"的顺序触发了问题，真实流程不受影响

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); cpu=lambda mod: [n+'.'+k for n,m in mod.named_modules() for k,v in list(vars(m).items())+list(m._parameters.items())+list(m._buffers.items()) if torch.is_tensor(v) and v.device.type=='cpu']; c=C(num_classes=2).cuda(); x=torch.randn(4,22,1280,device='cuda'); c.train(); c(x.unsqueeze(1)); c.eval(); a=A(num_classes=2, classification_type='bc').cuda().eval(); print('CPU tensors in a BEFORE load:', cpu(a)); a.load_state_dict(c.state_dict(), strict=True); print('CPU tensors in a AFTER load:', cpu(a)); a2=A(num_classes=2, classification_type='bc'); a2.load_state_dict({k: v.cpu() for k,v in c.state_dict().items()}, strict=True); a2=a2.cuda().eval(); print('CPU tensors in a2 (load on CPU, then .cuda()):', cpu(a2)); exec('try:\n a2(x, None); print(\"a2 forward: OK\")\nexcept Exception as e:\n print(\"a2 forward FAILED:\", type(e).__name__)')" 2>&1 | grep -E "^(CPU tensors|a2 forward)|Error"
```

**诊断 2 结果（2026-09-27）**：

| 情况 | 留在 CPU 上的张量 | 前向 |
|---|---|---|
| `a` 加载前（已 `.cuda()`） | `[]` | — |
| `a` **先 `.cuda()` 再加载** | 每个激活量化层 4 个：`scaling_impl.value`、`scaling_impl.buffer`、`zero_point_impl.zero_point.value`、`msb_clamp_bit_width_impl.bit_width.value`（共约 140 个） | ❌ 设备不一致 |
| `a2` **先在 CPU 上加载、再 `.cuda()`** | `[]` | ✅ `a2 forward: OK` |

**结论**：模型已经在 GPU 上时调用 `load_state_dict`，Brevitas 会在加载过程中**重新创建**激活量化层的这几个内部张量，默认放在 CPU 上。
它们是正常注册的参数/buffer，所以加载之后再 `.cuda()` 一次就会被一起搬到 GPU。**适配类本身没有问题。**

**真实流程不受影响**：

- 微调加载预训练权重（`FinetuneTask.load_pretrained_checkpoint`）发生在 `trainer.fit` 之前，此时模型还在 CPU 上；之后 Lightning 才把模型搬到 GPU —— 正好是安全顺序。
- QAP 冒烟训练最后的 `final_validate` 能成功也是这个原因：Lightning 在 fit 结束时会把模型搬回 CPU，再加载 checkpoint、再搬到 GPU
  （之前 `weights_only` 报错的调用栈里 `_restore_modules_and_callbacks` 就在 `strategy.setup` 之前）。

**注意事项**：自己写脚本加载 Brevitas 模型时，要**先在 CPU 上加载、再 `.cuda()`**（或加载后再调一次 `.cuda()`）。

### 最终版验证 A（加载顺序改为：先加载，再 `.cuda()`）

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "^(value|strict|logits|returns|mc )|Error"
import torch
from models.FEMBA_int8 import FEMBATinyInt8Classifier as A
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C

torch.manual_seed(0)
c = C(num_classes=2).cuda()
x = torch.randn(4, 22, 1280, device="cuda")
nv = lambda m: sum("scaling_impl.value" in k for k in m.state_dict())   # 数一下 state_dict 里有几个激活 scale

print("value keys before warmup:", nv(c))          # 期望 0：还没初始化，导出时被跳过
c.train(); c(x.unsqueeze(1)); c.eval()             # warmup：训练模式跑一次前向，初始化所有激活 scale
print("value keys after warmup:", nv(c))           # 期望 35

a = A(num_classes=2, classification_type="bc")     # 先留在 CPU
a.load_state_dict(c.state_dict(), strict=True)     # 在 CPU 上严格加载（Brevitas 会在 CPU 上重建激活量化的内部张量）
a = a.cuda().eval()                                # 加载完再搬到 GPU，内部张量一起搬过去
print("strict load: OK")

mask = torch.zeros_like(x, dtype=torch.bool)       # FinetuneTask 传的就是全 False 的假 mask
logits_a, x_out = a(x, mask)                       # 适配类：forward(x, mask) -> (logits, x)
logits_c = c(x.unsqueeze(1))                       # 原版：forward([B,1,H,W]) -> logits
print("logits shape", tuple(logits_a.shape), "| identical:", torch.equal(logits_a, logits_c))
print("returns input unchanged:", x_out is x)

try:                                               # 不支持的分类类型应该直接报错
    A(classification_type="mc")
    print("mc NOT rejected")
except NotImplementedError:
    print("mc rejected: OK")
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C(num_classes=2).cuda(); x=torch.randn(4,22,1280,device='cuda'); nv=lambda m: sum('scaling_impl.value' in k for k in m.state_dict()); print('value keys before warmup:', nv(c)); c.train(); c(x.unsqueeze(1)); c.eval(); print('value keys after warmup:', nv(c)); a=A(num_classes=2, classification_type='bc'); a.load_state_dict(c.state_dict(), strict=True); a=a.cuda().eval(); print('strict load: OK'); mask=torch.zeros_like(x,dtype=torch.bool); la,xo=a(x,mask); lc=c(x.unsqueeze(1)); print('logits shape', tuple(la.shape), '| identical:', torch.equal(la,lc)); print('returns input unchanged:', xo is x); exec('try:\n A(classification_type=\"mc\"); print(\"mc NOT rejected\")\nexcept NotImplementedError:\n print(\"mc rejected: OK\")')" 2>&1 | grep -E "^(value|strict|logits|returns|mc )|Error"
```

**预期**：

```
value keys before warmup: 0
value keys after warmup: 35
strict load: OK
logits shape (4, 2) | identical: True
returns input unchanged: True
mc rejected: OK
```

**结果（2026-09-27）**：✅ 全部符合预期

```
value keys before warmup: 0
value keys after warmup: 35
strict load: OK
logits shape (4, 2) | identical: True
returns input unchanged: True
mc rejected: OK
```

适配类跟原版 `FEMBATinyInt8` 结构完全一致（严格加载成功）、计算逐位相同，接口转换正确，不支持的 `mc` 被拒绝。

---

## 3. 验证 B：加载真实的 QAP 预训练 checkpoint

用第 5、6 步冒烟训练**实际生成的** checkpoint（`$CHECKPOINT_DIR/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt`，取最新的一个），
按 `FinetuneTask` 的方式加载进适配类，检查：

- 编码器权重**全部**加载上（`missing` 只能是分类头）
- 预训练多出来的只有 Decoder（`unexpected` 只能是 `decoder.*`）
- 权重值真的被复制过来了（抽查 SSM 核心参数 `A_log`）
- 加载后能正常前向

说明：checkpoint 里的参数名带 `model.` 前缀（`MaskTask.model`）；`FinetuneTask` 的模型也叫 `self.model`，所以它在 LightningModule 层面加载时前缀能对上。
这里直接加载到模型上，所以先去掉 `model.` 前缀，效果相同。

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF' 2>&1 | grep -E "^(ckpt|missing|unexpected|A_log|logits)|Error"
import glob, os, torch
from models.FEMBA_int8 import FEMBATinyInt8Classifier as A

# 找到冒烟训练生成的最新 checkpoint（跳过 last.ckpt，取按 val_loss 保存的那个）
paths = glob.glob("/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt")
path = max(paths, key=os.path.getmtime)
print("ckpt:", path)

ck = torch.load(path, map_location="cpu", weights_only=False)   # 自己训练出来的 checkpoint，来源可信
sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}   # 去掉 "model." 前缀

a = A(num_classes=2, classification_type="bc")
res = a.load_state_dict(sd, strict=False)          # 分类头预训练里没有、Decoder 微调里不要，所以用 strict=False
print("missing:", res.missing_keys)                # 期望只有 classifier.* 和 pre_classifier_quant.*
print("unexpected (non-decoder):", [k for k in res.unexpected_keys if not k.startswith("decoder.")])   # 期望 []

key = "mamba_blocks.0.mamba_fwd.ssm.A_log"         # 抽查一个 SSM 核心参数，确认值真的被复制过来了
print("A_log copied:", torch.equal(a.state_dict()[key], sd[key]))

a = a.cuda().eval()
logits, _ = a(torch.randn(4, 22, 1280, device="cuda"), None)
print("logits after load:", tuple(logits.shape), "finite:", bool(torch.isfinite(logits).all()))
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import glob, os, torch; from models.FEMBA_int8 import FEMBATinyInt8Classifier as A; path=max(glob.glob('/root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/*/epoch=*.ckpt'), key=os.path.getmtime); print('ckpt:', path); ck=torch.load(path, map_location='cpu', weights_only=False); sd={k[len('model.'):]: v for k,v in ck['state_dict'].items() if k.startswith('model.')}; a=A(num_classes=2, classification_type='bc'); res=a.load_state_dict(sd, strict=False); print('missing:', res.missing_keys); print('unexpected (non-decoder):', [k for k in res.unexpected_keys if not k.startswith('decoder.')]); key='mamba_blocks.0.mamba_fwd.ssm.A_log'; print('A_log copied:', torch.equal(a.state_dict()[key], sd[key])); a=a.cuda().eval(); lg,_=a(torch.randn(4,22,1280,device='cuda'), None); print('logits after load:', tuple(lg.shape), 'finite:', bool(torch.isfinite(lg).all()))" 2>&1 | grep -E "^(ckpt|missing|unexpected|A_log|logits)|Error"
```

**预期**：

```
ckpt: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_.../epoch=0-step=20.ckpt
missing: ['pre_classifier_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value', 'classifier.weight', 'classifier.bias']
unexpected (non-decoder): []
A_log copied: True
logits after load: (4, 2) finite: True
```

**结果（2026-09-27）**：✅ 全部符合预期

```
ckpt: /root/autodl-tmp/experiments/checkpoints/FEMBA_qap_demo/FEMBA_qap_demo_27_09_05-31-02.549489/epoch=0-step=20.ckpt
missing: ['pre_classifier_quant.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value', 'classifier.weight', 'classifier.bias']
unexpected (non-decoder): []
A_log copied: True
logits after load: (4, 2) finite: True
```

QAP 预训练 checkpoint 里的编码器权重（含所有激活量化 scale）**全部**加载进适配类；只缺分类头 3 个参数（微调时从头训练），只多出 Decoder（丢弃）。

---

## 本步结论 ✅

`FEMBATinyInt8Classifier` 完成，解决了问题 #2（接口对不上）：

- 结构、计算跟 ARES 原版 `FEMBATinyInt8` 完全一致（验证 A）
- QAP 预训练权重能完整加载（验证 B）
- 发现并记录了一个 Brevitas 的坑：GPU 上的模型调 `load_state_dict` 会在 CPU 上重建激活量化内部张量 —— 先在 CPU 加载、再 `.cuda()`；真实 Lightning 流程天然是这个顺序

剩余问题：#1 `FEMBA_quantized.yaml` 配置、#3 `finetune_task.py` 的 `weights_only`、#4 带标签的假数据。
