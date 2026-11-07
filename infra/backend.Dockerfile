FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 60 --retries 6 torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu
COPY backend/pyproject.toml ./
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 60 --retries 6 . \
    && useradd --create-home photonica
COPY artifacts/doe-unet.pt ./artifacts/doe-unet.pt
USER photonica
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "doe.api:app", "--host", "0.0.0.0", "--port", "8000"]
