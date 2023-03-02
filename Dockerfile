FROM continuumio/miniconda3
RUN conda env create -f ./alma_prox_segmentation/seg_attacks_env.yml
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
