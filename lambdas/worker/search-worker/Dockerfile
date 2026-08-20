# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN dnf -y install gcc gcc-c++ make git && dnf clean all

COPY lambdas/worker/search-worker/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY shared ${LAMBDA_TASK_ROOT}/shared
COPY lambdas ${LAMBDA_TASK_ROOT}/lambdas
COPY lambdas/worker/search-worker/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.handler"]
