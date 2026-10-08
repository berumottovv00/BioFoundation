# FEMBA INT8 QAP 验证命令

验证 `models/FEMBA_int8.py`（`FEMBATinyInt8Pretrain`）是否正确。所有命令都在 AutoDL 服务器上运行。

每一步都给了两种写法：

- **带注释的脚本**：用来阅读理解；粘贴时如果终端给每行前面自动加了空格，`EOF` 会失效，Python 也会报缩进错误。
- **单行命令**：内容完全相同，没有注释，直接复制最稳。

---

## 0. 环境检查

```bash
cd /root/autodl-tmp/BioFoundation      # 进入项目根目录，models. 和 ARES. 的导入都从这里开始找
nvidia-smi --query-gpu=name --format=csv,noheader   # 显卡型号；必须是 Blackwell 之前的卡（4090 可以，5090 不行）

python - <<'EOF'
import torch
print(torch.__version__)                    # 期望 2.7.1+cu126
print(torch.cuda.get_device_name(0))        # 显卡名称
a = torch.randn(64, 64, device='cuda')
b = torch.randn(64, 64, device='cuda')
print((a @ b).sum().item())                 # 能打印出一个数字，说明 GPU 上能正常计算
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && nvidia-smi --query-gpu=name --format=csv,noheader && python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), (torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum().item())"
```

**结果（2026-09-26，RTX 4090）**：`2.7.1+cu126 NVIDIA GeForce RTX 4090 185.21...` ✅

---

## 1. 第 1 层：模型能不能正常使用

检查四件事：输出形状、梯度能不能传到参数、权重能不能加载到原版分类模型、有没有导入 mamba_ssm。

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF'
import sys, torch
from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P                          # P：QAP 预训练模型
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C   # C：原版量化分类模型

m = P().cuda()                                         # 创建预训练模型，放到 GPU
x = torch.randn(2, 22, 1280, device='cuda')            # 假输入：2 个样本 × 22 通道 × 1280 个采样点
mask = torch.zeros_like(x, dtype=torch.bool)           # 掩码，先全部设为 False
mask[:, :2, :16] = True                                # 只遮住左上角一个 patch（2 通道 × 16 个点）

r, o = m(x, mask)                                      # 前向：r = 重建信号，o = 原始信号
print('shapes', tuple(r.shape), tuple(o.shape))        # 期望 (2, 22, 1280) (2, 22, 1280)

r[mask].pow(2).mean().backward()                       # 用被遮住位置构造一个损失，反向传播
print('params with grad',
      sum(p.grad is not None for p in m.parameters()), '/',
      sum(1 for p in m.parameters()))                  # 有梯度的参数数 / 参数总数

res = C().load_state_dict(m.state_dict(), strict=False)   # 把预训练权重加载到原版分类模型
print('missing', res.missing_keys)                     # 分类模型有、预训练模型没有的：应该只有分类头
print('unexpected', [k for k in res.unexpected_keys
                     if not k.startswith('decoder.')]) # 预训练模型多出来的（去掉 decoder）：应该是 []
print('mamba_ssm loaded:', 'mamba_ssm' in sys.modules) # 应该是 False：QAP 不依赖 mamba_ssm
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import sys, torch; from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; m=P().cuda(); x=torch.randn(2,22,1280,device='cuda'); mask=torch.zeros_like(x,dtype=torch.bool); mask[:,:2,:16]=True; r,o=m(x,mask); print('shapes', tuple(r.shape), tuple(o.shape)); r[mask].pow(2).mean().backward(); print('params with grad', sum(p.grad is not None for p in m.parameters()), '/', sum(1 for p in m.parameters())); res=C().load_state_dict(m.state_dict(), strict=False); print('missing', res.missing_keys); print('unexpected', [k for k in res.unexpected_keys if not k.startswith('decoder.')]); print('mamba_ssm loaded:', 'mamba_ssm' in sys.modules)"
```

**结果（2026-09-26）**：

| 检查项 | 结果 |
|---|---|
| shapes | `(2, 22, 1280) (2, 22, 1280)` ✅ |
| params with grad | `80 / 84` ⚠️ 有 4 个参数没有梯度，见第 3 步 |
| missing | `pre_classifier_quant...scaling_impl.value`、`classifier.weight`、`classifier.bias` ✅（只有分类头） |
| unexpected | `[]` ✅ |
| mamba_ssm loaded | `False` ✅ |

启动时打印的 `[FEMBATinyInt8] Configuration` 是原类自带的输出；`Named tensors ... experimental` 警告来自 Brevitas，都不影响结果。

---

## 2. 第 2 层：复制的编码器和原版数值上是否一致

`FEMBATinyInt8Pretrain.forward_features` 是从原版 `forward` 里逐行复制的编码器部分。
这里拿**同一个原版模型**，分别用"原版 forward"和"我们的 forward_features + 原版分类头"计算，结果必须逐位相同。

```bash
cd /root/autodl-tmp/BioFoundation

python - <<'EOF'
import torch
from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C

torch.manual_seed(0)                                   # 固定随机种子
c = C().cuda().eval()                                  # 原版模型；必须用 eval()，否则 Brevitas 每次前向
                                                       # 都会更新激活量化统计量，同一个输入两次结果也不同
x = torch.randn(2, 1, 22, 1280, device='cuda')         # 原版的输入形状是 [B, 1, H, W]

y = c(x)                                               # 路线 A：原版完整 forward → logits

f = P.forward_features(c, x)                           # 路线 B：用我们的 forward_features 跑原版模型的编码器
z = c.classifier(                                      #        再手动接上原版的分类头
        c.pre_classifier_quant(
            c.global_pool(f.transpose(1, 2)).squeeze(-1)))

print('features', tuple(f.shape))                      # 期望 (2, 80, 385)：80 个时间步 × 385 维
print('encoder identical:', torch.equal(y, z))         # 期望 True：两条路线逐位相同
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); c=C().cuda().eval(); x=torch.randn(2,1,22,1280,device='cuda'); y=c(x); f=P.forward_features(c,x); z=c.classifier(c.pre_classifier_quant(c.global_pool(f.transpose(1,2)).squeeze(-1))); print('features', tuple(f.shape)); print('encoder identical:', torch.equal(y,z))"
```

**结果（2026-09-27）**：`features (2, 80, 385)`、`encoder identical: True` ✅ —— 复制的编码器与原版逐位一致

---

## 3. 排查：哪 4 个参数没有梯度

找出第 1 步里没拿到梯度的参数名，并对原版分类模型做同样的检查。

- 两行列出的参数（除去分类头）一样 → 原版就是这样，不是我们引入的问题
- `PRETRAIN` 比 `ORIGINAL` 多出参数 → 是我们的改动导致的，需要修

```bash
cd /root/autodl-tmp/BioFoundation      # 进入项目根目录

python - <<'EOF' 2>&1 | grep -E "no-grad|Error"
# ↑ python - <<'EOF'：把下面到 EOF 为止的内容当作 Python 脚本执行
#   2>&1 | grep ...：把警告合并到输出里，然后只显示包含 "no-grad" 或 "Error" 的行

import torch
from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P                          # P：QAP 预训练模型
from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C   # C：原版量化分类模型

torch.manual_seed(0)                   # 固定随机种子，保证每次结果都一样

# ===== 检查 QAP 预训练模型 =====
m = P().cuda()                                     # 创建预训练模型，放到 GPU 上
x = torch.randn(2, 22, 1280, device='cuda')        # 假输入：2 个样本 × 22 通道 × 1280 个采样点
mask = torch.rand_like(x) < 0.6                    # 随机掩码：约 60% 的位置为 True（被遮住），跟真实训练的比例一致
r, o = m(x, mask)                                  # 前向计算：r = 重建信号，o = 原始信号
r[mask].pow(2).mean().backward()                   # 用被遮住位置的重建值构造一个损失，然后反向传播
                                                   # 参与了计算的参数会得到 .grad，没参与的仍然是 None
print('PRETRAIN no-grad:',
      [n for n, p in m.named_parameters() if p.grad is None])   # 打印没拿到梯度的参数名

# ===== 对照：检查原版分类模型 =====
c = C().cuda()                                     # 创建原版分类模型
c(x.unsqueeze(1)).sum().backward()                 # 原版要求输入是 [B,1,22,1280]，所以先 unsqueeze(1)
                                                   # 把输出的 logits 加起来当作损失，然后反向传播
print('ORIGINAL no-grad:',
      [n for n, p in c.named_parameters() if p.grad is None])   # 打印原版里没拿到梯度的参数名
EOF
```

单行版：

```bash
cd /root/autodl-tmp/BioFoundation && python -c "import torch; from models.FEMBA_int8 import FEMBATinyInt8Pretrain as P; from ARES.tests.test_networks.test_25_femba_tiny_int8 import FEMBATinyInt8 as C; torch.manual_seed(0); m=P().cuda(); x=torch.randn(2,22,1280,device='cuda'); mask=torch.rand_like(x)<0.6; r,o=m(x,mask); r[mask].pow(2).mean().backward(); print('PRETRAIN no-grad:', [n for n,p in m.named_parameters() if p.grad is None]); c=C().cuda(); c(x.unsqueeze(1)).sum().backward(); print('ORIGINAL no-grad:', [n for n,p in c.named_parameters() if p.grad is None])" 2>&1 | grep -E "no-grad|Error"
```

**结果（2026-09-27）**：两边完全相同 ✅

```
PRETRAIN no-grad: ['mamba_blocks.{0,1}.mamba_{fwd,rev}.conv1d.weight_scale']  （共 4 个）
ORIGINAL no-grad: ['mamba_blocks.{0,1}.mamba_{fwd,rev}.conv1d.weight_scale']  （共 4 个）
```

原因：`QuantConv1dDepthwise`（`ARES/tests/test_networks/brevitas_custom_layers.py`）里
`self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)` 是作者有意设成不可训练的占位参数，
forward 里也没有用到。原版就是这样，不是 QAP 改动引入的，对训练没有影响。

### 已知限制：深度卷积的权重在训练时没有伪量化

`QuantConv1dDepthwise` 内部用的是普通 `nn.Conv1d`，训练时只量化了输出，**权重保持 FP32**；
ARES 导出时（`ARES/tools/pytorch_extractor.py` 的 `_extract_conv1d_depthwise_info`）才把权重事后量化成 INT8。
因此这一层的权重量化误差在 QAP 训练中感受不到，部署时才出现。

- 影响范围：每个 Mamba 方向 1540 × 4 个权重，占全模型约 760 万参数的很小一部分
- 原版量化微调流程同样存在这个问题
- 处理方式：暂不修改；等 QAP 全流程跑通后，根据部署精度损失再决定是否把这层换成量化卷积（需同时确认 ARES 导出支持）
