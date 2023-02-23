#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python attack_experiment.py -F ../../test/test500 with dataset.cityscapes cudnn_flag=benchmark attack.dag attack.gamma=0.001 target=0 dataset.num_images=500

