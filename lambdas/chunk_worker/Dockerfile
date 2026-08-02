# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for NLP wheels that may build native extensions.
RUN dnf -y install gcc gcc-c++ make git && dnf clean all

COPY lambdas/chunk_worker/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Pre-download GLiNER model into the image so Lambda never needs network access
# for the model weights. HF_HOME points to a path inside the image that is
# readable at runtime. HF_HUB_OFFLINE=1 is set at Lambda runtime (not here)
# to prevent cache-write attempts on the read-only /var/task filesystem.
ENV HF_HOME=/var/task/models
RUN python3 -c "\
from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_mediumv2.1', cache_dir='/var/task/models')"

# Copy runtime code
COPY shared ${LAMBDA_TASK_ROOT}/shared
COPY lambdas ${LAMBDA_TASK_ROOT}/lambdas
COPY lambdas/chunk_worker/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.handler"]
