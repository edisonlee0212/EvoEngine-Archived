FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as base

RUN apt update && \
    apt install ninja-build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    htop \
    wget \ 
    python3.10 \
    python3-pip \
    sudo \
    nano \
    ninja-build \
    libassimp-dev \
    libglm-dev \
    libglfw3-dev \
    libxi-dev \
    libxrandr-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
RUN sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.261-jammy.list https://packages.lunarg.com/vulkan/1.3.261/lunarg-vulkan-1.3.261-jammy.list
RUN sudo apt update
RUN apt install -y vulkan-sdk
RUN apt install -y unzip

RUN git clone --recursive https://github.com/edisonlee0212/EvoEngine.git /home/root/evoengine
WORKDIR /home/root/evoengine/3rdParty/shaderc
RUN cat ./shaderc.zip* > ./shaderc.zip
RUN unzip ./shaderc.zip
RUN rm ./shaderc.zip

WORKDIR /home/root/evoengine/3rdParty/physx
RUN cat ./physx.zip* > ./physx.zip
RUN unzip ./physx.zip
RUN rm ./physx.zip

WORKDIR /home/root/evoengine
RUN bash ./build.sh

WORKDIR /home/root/
ENTRYPOINT [  "/bin/bash" ]


RUN bash