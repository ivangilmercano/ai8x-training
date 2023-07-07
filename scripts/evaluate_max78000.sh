#!/bin/sh
MODEL="ai85cdnet"
DATASET="cats_vs_dogs"
QUANTIZED_MODEL="../ai8x-training/logs/2023.03.24-201120_cats_vs_dogs/qat_best-q.pth.tar"

# evaluate scripts for cats vs dogs
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL  -8 --save-sample 1 --device MAX78000 "$@"

#evaluate scripts for kws
# python train.py --model $MODEL --dataset $DATASET --confusion --evaluate --exp-load-weights-from $QUANTIZED_MODEL -8 --save-sample 1 --device MAX78000 "$@"

