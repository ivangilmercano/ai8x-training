#!/bin/sh
python train.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-training/logs/2023.03.24-201120_cats_vs_dogs/qat_best-q.pth.tar --save-sample 1 -8 --device MAX78000 "$@"
