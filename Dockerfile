FROM continuumio/miniconda3
RUN conda create -n env python=3.9
RUN echo "source activate env" > ~/.bashrc
RUN ./install.sh
ENV PATH /opt/conda/envs/env/bin:$PATH
