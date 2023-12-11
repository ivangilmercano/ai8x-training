#!/bin/sh
python train.py --deterministic --epochs 250 --batch-size 100 --optimizer Adam --lr 0.001 --wd 0 --qat-policy policies/qat_policy_cakes.yaml --compress policies/schedule-cakes.yaml --model ai85cdcake --dataset cakes --device MAX78000 --print-freq 100 --validation-split 0 --confusion "$@"
