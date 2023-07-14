#!/bin/sh
python train.py --epochs 20 --batch-size 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20mlpv3 --dataset kws20_mlp --confusion --device MAX78000 "$@"
