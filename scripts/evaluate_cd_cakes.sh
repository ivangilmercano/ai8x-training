#!/bin/sh
MODEL="ai85cdcake"
DATASET="cakes"
QUANTIZED_MODEL="../ai8x-training/logs/cd-cakes/best-q.pth.tar"

# evaluate scripts for cats vs dogs
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"
