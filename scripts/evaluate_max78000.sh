#!/bin/sh
MODEL="ai85kws20netv3"
DATASET="KWS_20"
QUANTIZED_MODEL="../ai8x-training/logs/2023.04.06-172201_kws/qat_best-q.pth.tar"

# evaluate scripts for cats vs dogs
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 1 --device MAX78000 "$@"

#evaluate scripts for kws
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

