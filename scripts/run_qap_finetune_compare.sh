#!/usr/bin/env bash
# QAP vs. from-scratch INT8 fine-tuning on TUAR (binary), run back to back.
# Only difference between the two runs: whether the QAP-pretrained encoder is loaded.
# See docs/08_qap_finetune_compare.md.
#
#   nohup bash scripts/run_qap_finetune_compare.sh > /root/autodl-tmp/qap_ft_compare.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

export DATA_PATH=${DATA_PATH:-/root/autodl-tmp/data}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/root/autodl-tmp/experiments}
QAP_CKPT=${QAP_CKPT:-$CHECKPOINT_DIR/checkpoints/FEMBA_qap_tuar/FEMBA_qap_tuar_27_09_16-22-31.239854/epoch=9-step=15610.ckpt}
LOG_DIR=${LOG_DIR:-/root/autodl-tmp}

if [ ! -f "$QAP_CKPT" ]; then
  echo "QAP checkpoint not found: $QAP_CKPT"
  exit 1
fi

# Shared by both runs (FEMBA_quantized.yaml is written for 4-GPU DDP, batch 256)
COMMON=(
  +experiment=FEMBA_quantized
  data_module=finetune_data_module_tuar       # TUAR instead of TUAB
  gpus=1
  trainer.strategy=auto                       # single GPU, no DDP
  batch_size=32
  num_workers=4
  scheduler.warmup_epochs=1                   # default 5 == max_epochs would keep the whole run in warmup
  io.base_output_path=$CHECKPOINT_DIR/tb_logs
)

echo "[$(date '+%F %T')] B: QAP-pretrained encoder ($QAP_CKPT)"
python -u run_train.py "${COMMON[@]}" tag=FEMBA_qap_ft_B_qap \
  "pretrained_checkpoint_path='$QAP_CKPT'" > "$LOG_DIR/qap_ft_B_qap.log" 2>&1
echo "[$(date '+%F %T')] B finished, exit code $?"

echo "[$(date '+%F %T')] A: from scratch (random init)"
python -u run_train.py "${COMMON[@]}" tag=FEMBA_qap_ft_A_scratch \
  pretrained_checkpoint_path=null > "$LOG_DIR/qap_ft_A_scratch.log" 2>&1
echo "[$(date '+%F %T')] A finished, exit code $?"
