#!/bin/sh

# evaluate scripts for cats vs dogs
python train.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-training/logs/2023.03.24-201120_cats_vs_dogs/qat_best-q.pth.tar -8 --save-sample 1 --device MAX78000 "$@"
python move/move_sample_cats_dogs.py

#evaluate scripts for kws
# python train.py --model ai85kws20netv3 --dataset KWS_20 --confusion --evaluate --exp-load-weights-from ../ai8x-training/logs/2023.04.06-172201_kws/qat_best-q.pth.tar -8 --save-sample 1 --device MAX78000 "$@"
# python move/move_sample_kws.py
# 2023.04.06-172201_kws