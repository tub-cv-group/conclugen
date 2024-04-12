FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update &&\
    apt-get install wget -y

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
RUN rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN . /opt/miniconda3/etc/profile.d/conda.sh
ENV PATH="/opt/miniconda3/bin:$PATH"

RUN touch ~/.bashrc

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6 libxdamage-dev git build-essential libedit-dev -y

RUN conda update conda
RUN conda install python=3.10
COPY environment.yaml .
RUN conda env update -n base --file environment.yaml

ENV  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/:/opt/miniconda3/lib"

ENTRYPOINT [ "/bin/bash", "-c" ]