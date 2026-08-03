# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN dnf -y install gcc gcc-c++ make git && dnf clean all

COPY lambdas/search_worker/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Bake BAAI/bge-reranker-base into the image at build time.
# At Lambda runtime the reranker copies this to /tmp before loading
# (same pattern as GLiNER in the NER lambda).
ENV HF_HOME=/var/task/models
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')" && \
    find /var/task/models -type f -exec chmod a+r {} + && \
    find /var/task/models -type d -exec chmod a+rx {} +

COPY shared ${LAMBDA_TASK_ROOT}/shared
COPY lambdas ${LAMBDA_TASK_ROOT}/lambdas
COPY lambdas/search_worker/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.handler"]
