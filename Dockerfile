FROM continuumio/miniconda3
ADD ./seg_attacks_env.yml seg_attacks_env.yml
RUN echo $(ls)
RUN conda env create -f seg_attacks_env.yml
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
