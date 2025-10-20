FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir numpy

# your pinned stack + boto3 to match botocore 1.37.35
RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-hcr-data-transformation.git@v0.0.5 \
    aind-data-schema==1.4.0 \
    botocore==1.37.35 \
    boto3==1.37.35 \
    awscli==1.38.35 \
    numcodecs==0.13.1 \
    numba==0.61.2 \
    tensorstore==0.1.72 \
    s3fs==0.4.2 \
    jupyterlab

WORKDIR /app
COPY Rhapso /app/Rhapso
ENV PYTHONPATH=/app
CMD ["python", "-m", "Rhapso.fusion.run_multiscale"]


