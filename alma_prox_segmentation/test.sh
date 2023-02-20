#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python attack_experiment.py -F ../../test with dataset.cityscapes cudnn_flag=benchmark attack.alma_prox_linf target=0
