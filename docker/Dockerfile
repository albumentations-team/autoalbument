FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --no-log-init autoalbument
USER autoalbument
ENV PATH="/home/autoalbument/.local/bin:${PATH}"
WORKDIR /opt/autoalbument
COPY ./docker/requirements.txt /opt/autoalbument/docker/requirements.txt
RUN pip install --no-cache-dir -r /opt/autoalbument/docker/requirements.txt
RUN pip install --no-cache-dir https://github.com/albumentations-team/albumentations/archive/random_crop_and_pad.zip
COPY . .
RUN pip install --no-cache-dir .
COPY docker/entrypoint.sh entrypoint.sh

WORKDIR /autoalbument

ENTRYPOINT ["/opt/autoalbument/entrypoint.sh"]
