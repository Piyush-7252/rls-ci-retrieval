# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for NLP wheels that may build native extensions.
RUN dnf -y install gcc gcc-c++ make git && dnf clean all

COPY lambdas/worker/document-chunk-worker/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Bake GLiNER model + deberta tokenizer into the image at build time.
# HF_HOME controls where huggingface_hub writes the cache (hub/models--*/snapshots/...).
# Do NOT pass cache_dir= to from_pretrained — it conflicts with HF_HOME and produces
# an incomplete snapshot that fails to load at runtime.
# At Lambda runtime the NER lambda copies /var/task/models → /tmp/hf_cache (writable)
# and sets HF_HOME=/tmp/hf_cache before calling GLiNER.from_pretrained().
# chmod is in the same RUN step so model files and permissions are ONE layer (not two).
ENV HF_HOME=/var/task/models
RUN python3 -c "\
from gliner import GLiNER; \
GLiNER.from_pretrained('urchade/gliner_mediumv2.1'); \
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); \
print('GLiNER + deberta tokenizer baked OK')" && \
    find /var/task/models -type f -exec chmod a+r {} + && \
    find /var/task/models -type d -exec chmod a+rx {} +

# Copy runtime code
COPY shared ${LAMBDA_TASK_ROOT}/shared
COPY lambdas ${LAMBDA_TASK_ROOT}/lambdas
COPY lambdas/worker/document-chunk-worker/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.handler"]
