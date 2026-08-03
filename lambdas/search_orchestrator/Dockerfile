# syntax=docker/dockerfile:1.7
FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY lambdas/search_orchestrator/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# No ML models — pure orchestration logic.
COPY shared ${LAMBDA_TASK_ROOT}/shared
COPY lambdas ${LAMBDA_TASK_ROOT}/lambdas
COPY lambdas/search_orchestrator/lambda_function.py ${LAMBDA_TASK_ROOT}/lambda_function.py

CMD ["lambda_function.handler"]
