FROM python:3.8.6

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /autoalbument

COPY tests_e2e/requirements.txt /autoalbument/tests_e2e/requirements.txt
RUN pip install --no-cache-dir -r /autoalbument/tests_e2e/requirements.txt

COPY . .
RUN pip install /autoalbument

ENTRYPOINT ["/autoalbument/tests_e2e/run.sh"]
