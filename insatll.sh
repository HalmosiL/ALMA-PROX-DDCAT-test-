#!/bin/bash

PATH="/home/halmosi/"

#Clone Repository
git clone https://github.com/HalmosiL/ALMA-PROX-DDCAT-test-.git

#Create
mkdir data
mkdir test
mkdir models 

#Create docker image
cd ALMA-PROX-DDCAT-test-
docker build -t alma-prox .

nvidia-docker run -it --gpus all \
--mount type=bind,source='${PATH}ALMA-PROX-DDCAT-test-/alma_prox_segmentation/,target=/alma_prox_segmentation/' \
--mount type=bind,source='${PATH}data/,target=/alma_prox_segmentation/data/cityscapes/' \
--mount type=bind,source='${PATH}test/,target=/test/' \
--mount type=bind,source='${PATH}models/,target=/models/' \
alma-prox
