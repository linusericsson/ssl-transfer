FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime
RUN apt-get update && apt-get -y install gcc
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --requirements git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git
