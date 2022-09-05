FROM registry.datexis.com/bwinter/datexis-pytorch:python3.7-cuda11.0


ENV PYTHONIOENCODING=utf-8
RUN pip install tqdm==4.64.0
#  - python==3.8.13
RUN pip install torch
RUN pip install torchvision==0.12.0
RUN pip install torchmetrics==0.9.3

COPY . /src

WORKDIR /src