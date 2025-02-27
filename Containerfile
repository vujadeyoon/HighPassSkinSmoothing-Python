# Dveloper: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon
# Personal website: https://vujadeyoon.github.io
#
# Title: Containerfile
# Description: A Containerfile for the NVIDIA Container Toolkit for Deep Learning
#
#
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


LABEL maintainer="vujadeyoon"
LABEL email="vujadeyoon@gmial.com"
LABEL version="1.0"
LABEL description="Docker with NVIDIA Container Toolkit for Deep Learning"


ARG UID="1000"
ARG GID="1000"
ARG GIT_MAINTAINER="vujadeyoon"
ARG GIT_EMAIL="vujadeyoon@gmail.com"
ARG PATH_GIT_PROJ="/home/dev/proj"


ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_MAINTAINER=${GIT_MAINTAINER}
ENV GIT_EMAIL=${GIT_EMAIL}
ENV PATH_GIT_PROJ=${PATH_GIT_PROJ}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ Asia/Seoul


# Fix a bug related to the below messages:
#     i) W: GPG error: http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64
#           InRelease: The following signatures couldn't be verified because the public key is not available:
#           NO_PUBKEY A4B469963BF863CC.
#     ii) W: The repository 'http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64
#            InRelease' is not signed.
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC


# Make required directories.
RUN mkdir -p /home/dev/ && \
    mkdir -p /home/python_packages/ && \
    mkdir -p /home/deb_packages/


# Install the essential ubuntu packages.
RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        cmake \
        curl \
        vim \
        ssh \
        sudo \
        tar \
        libcurl3-dev \
        libfreetype6-dev \
        pkg-config \
        ca-certificates \
        libjpeg-dev \
        libpng-dev


# Install the useful ubuntu packages.
RUN apt-get update &&  \
    apt-get install -y \
        eog \
        nautilus \
        imagemagick \
        libreoffice \
        python3-tk \
        pv \
        dialog \
        ffmpeg \
        libgtk2.0-dev \
        python3-matplotlib \
        wget \
        tmux \
        zsh \
        locales \
        ncdu \
        htop \
        zip \
        unzip \
        rsync


# Install git.
RUN apt-get update && \
    apt-get install -y git && \
    git config --global user.name "${GIT_MAINTAINER}" && \
    git config --global user.email "${GIT_EMAIL}" && \
    git config --global --add safe.directory "${PATH_GIT_PROJ}"


# Install editors.
RUN apt-get update && \
    apt-get install -y vim && \
    echo "set number" | tee -a ~/.vimrc && \
    echo "set ts=8" | tee -a ~/.vimrc && \
    echo "set sw=4" | tee -a ~/.vimrc && \
    echo "set sts=4" | tee -a ~/.vimrc && \
    echo "set smartindent" | tee -a ~/.vimrc && \
    echo "set cindent" | tee -a ~/.vimrc


# Set the locale.
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Install Korean language.
RUN apt-get update &&  \
    apt-get install -y \
        fcitx \
        fcitx-hangul \
        fonts-nanum*


# Install python.
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3-pip


# Set python3 configurations.
# RUN apt-get update && \
#    apt-get install -y \
#        python3.9 \
#        python3.9-dev && \
#        python3.9-venv && \
#    rm -f /usr/bin/python3 && \
#    ln -s /usr/bin/python3.7 /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python



# Install more python packages.
RUN pip install --upgrade pip && \
    pip3 install --upgrade pip


# Python3 packages for mathematical functions and plotting
RUN pip3 install \
    opencv-python \
    opencv-contrib-python \
    ffmpeg-python \
    Pillow \
    imageio \
    matplotlib \
    scikit-image \
    scikit-learn \
    pandas \
    openpyxl \
    plotly \
    seaborn \
    shapely


# Python3 packages for monitoring and debugging
RUN pip3 install \
    jupyter \
    wandb \
    gpustat \
    getgpu \
    tqdm \
    ipdb \
    icecream \
    progressbar


# Other python3 packages
RUN pip3 install \
    scipy \
    Cython \
    prettyprinter \
    colorlog \
    randomcolor \
    future==0.18.2 \
    imutils \
    psutil \
    PyYAML \
    pycrypto


# Install python3 packages related to the server.
RUN pip install --upgrade --ignore-installed blinker
RUN pip3 install \
    Flask \
    Flask-RESTful \
    gevent \
    boto3 \
    kubernetes


# Install python3 packages for the deep learning research.
RUN pip install --upgrade --ignore-installed sympy
RUN pip3 install \
    dlib \
    PyWavelets \
    pycuda>=2020.1 \
    tensorflow \
    kornia \
    torchinfo \
    ptflops \
    onnx \
    onnxscript \
    onnxsim \
    onnxruntime
RUN pip3 install torch torchvision torchdata torchaudio torchtext --index-url https://download.pytorch.org/whl/cu121


# Update and clean the Ubuntu packages.
RUN apt-get update &&  \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*


# Enroll bash functions to ~/.bashrc.
COPY ./script/container/bash_enroll_func_bashrc.sh /home/dev/bash_enroll_func_bashrc.sh
RUN bash /home/dev/bash_enroll_func_bashrc.sh && \
    rm -rf /home/dev/bash_enroll_func_bashrc.sh


# Go to the base directory.
WORKDIR /home/dev/


# Run the command.
CMD [ "/bin/bash" ]
