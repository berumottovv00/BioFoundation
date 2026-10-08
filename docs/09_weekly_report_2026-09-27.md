# Weekly Report — 2026-09-27

## Main results

QAP (quantization-aware pretraining) for FEMBA-tiny INT8 now runs end to end: INT8 masked-reconstruction pretraining, then INT8 fine-tuning, then test. I pretrained with QAP on the TUAR train split (signals only, ~50k windows, 10 epochs, ~2 h on one GPU); the reconstruction loss was still decreasing at the end, so it hasn't converged. I then fine-tuned the INT8 model on TUAR binary classification (artifact vs. no artifact). On the test set, QAP initialization beat random initialization on every metric: AUROC +0.029, AUPR +0.025, Kappa +0.025. The gap is well beyond test-set sampling error (AUROC SE ≈ 0.004), but it comes from a single seed.

| | AUROC | AUPR | Acc | F1 | Kappa |
|---|---|---|---|---|---|
| B: QAP init | **0.901** | **0.918** | **0.816** | **0.820** | **0.633** |
| A: random init | 0.873 | 0.893 | 0.803 | 0.808 | 0.608 |

## Progress

I built the QAP pipeline on top of ARES's `FEMBATinyInt8`, a Brevitas INT8 version of FEMBA-tiny. It fake-quantizes the patch embedding, the bidirectional Mamba blocks (implemented in plain PyTorch, so no `mamba_ssm` is needed) and all activations, but it only had a classification head. For pretraining, I subclassed it, kept the INT8 encoder unchanged, removed the classifier and attached an FP32 convolutional decoder. The model is trained with FEMBA's masked reconstruction objective: 60% of 2×16 patches are masked, with Smooth-L1 loss on the masked positions. This way the encoder learns EEG representations under INT8 constraints from the start, and the decoder is discarded afterwards. For fine-tuning, I wrote a thin adapter subclass with the same parameter names, so the pretrained INT8 encoder, including its learned activation scales, loads directly and only a new INT8 classifier head is trained. I verified that the encoder output is bit-exact with the original ARES model and that all 81 encoder tensors transfer from pretraining to fine-tuning. The full pipeline is: QAP on unlabeled TUAR train signals, then INT8 fine-tuning on labeled TUAR, then test. It runs through the existing Hydra/Lightning tasks on a single AutoDL GPU. Getting it to run also required fixing broken quantized configs and several torch 2.7 incompatibilities in the repo.

## Problems

Each arm has only one seed, so the +0.029 AUROC gain may still be within training (seed) variance, which I haven't measured. The pretraining data is small and comes from the same dataset as the downstream task (the TUAR train split), so it is unclear how far the result generalizes. In the ARES INT8 model, the depthwise-conv weights are not fake-quantized during training; they are only quantized at export, so training and deployment don't fully match. The QAP pretraining validation loss is dominated by a few hard windows, likely strong artifacts (not yet verified). The input normalization is a fixed linear scaling rather than per-channel IQR, so artifact windows end up about 50× larger than normal EEG. Separately, a bug makes the `freeze_layers` option a no-op, so linear-probing evaluation isn't possible yet.

## Potential solutions

I will run 3 seeds per arm and report mean ± std, to separate a real effect from noise. I will pretrain longer, since the loss is still falling, and on more unlabeled EEG such as a TUEG subset, which gives QAP a fairer test. I will also replace the depthwise conv with a quantized conv, after checking that ARES export supports it, so that training matches deployment. For the hard windows, I will first compute per-window loss against amplitude to confirm the cause. I will then report the median validation loss alongside the mean, and test clipping the normalized input as a separate experiment, since amplitude is itself a useful artifact cue. For `freeze_layers`, I will fix the one-line bug and set the existing configs to `False` so their behavior doesn't change. I will also apply freezing to the random-init arm, so that a linear-probe comparison is possible.

## Plan for next week

First, finish the A vs. B comparison with 3 seeds per arm and report mean ± std. Second, extend QAP pretraining (more epochs and/or a TUEG subset) and re-run fine-tuning. Third, add a reference arm: FP pretraining followed by INT8 fine-tuning, or plain FP32 fine-tuning. Fourth, look into quantizing the depthwise conv and exporting the fine-tuned model with ARES. Fifth, commit the code and docs.
