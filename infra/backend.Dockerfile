FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home photonica
USER photonica
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "doe.api:app", "--host", "0.0.0.0", "--port", "8000"]
