-FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as build

# package requirements
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install build-essential git

# pip requirements
WORKDIR /opt/algorithm
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html \
    && conda clean -ya

# copy checkpoints
COPY checkpoints/mandibles.ckpt checkpoints/mandibles.ckpt
COPY checkpoints/condyles.ckpt checkpoints/condyles.ckpt

# copy Python files
COPY orbifrac/ orbifrac/
COPY predict_from_raw_data.py /opt/conda/lib/python3.10/site-packages/nnunetv2/inference/predict_from_raw_data.py



FROM nvidia/cuda:11.7.1-base-ubuntu20.04

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt -y --no-install-recommends install ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm \
    && mkdir /input /opt/algorithm /output \
    && chown -R algorithm:algorithm /input /opt/algorithm /output
USER algorithm

WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm --from=build /opt/conda/. /opt/conda
COPY --chown=algorithm:algorithm --from=build /opt/algorithm/. /opt/algorithm

# script to run
ENV nnUNet_raw=/opt/algorithm/nnUNet_raw
ENV nnUNet_preprocessed=/opt/algorithm/nnUNet_preprocessed
ENV nnUNet_results=/opt/algorithm/nnUNet_results
ENV OMP_NUM_THREADS=1
COPY process.py process.py
ENTRYPOINT ["/opt/conda/bin/python", "/opt/algorithm/process.py"]
