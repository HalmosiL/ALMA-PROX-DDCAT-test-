FROM continuumio/miniconda3
ADD ./alma_prox_segmentation/seg_attacks_env.yml seg_attacks_env.yml
RUN echo $(ls)
RUN conda env create -f seg_attacks_env.yml
RUN echo "source activate env" > ~/.bashrc
RUN conda activate seg_attacks
ENV PATH /opt/conda/envs/env/bin:$PATH
