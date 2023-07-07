#!/bin/sh
MODEL="ai85kws20mlpv2"
DATASET="KWS_20"
QUANTIZED_MODEL="../ai8x-training/logs/logs_kws-mlp-v4/best-quantized.pth.tar"

# evaluate scripts for kws mlp
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 6 --device MAX78000 "$@"
