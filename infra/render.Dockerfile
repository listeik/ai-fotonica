FROM node:24-alpine AS frontend
WORKDIR /app
RUN npm install --global pnpm@11.25.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir=/pnpm/store --network-concurrency=8 --fetch-retries=4
COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/src ./src
RUN pnpm build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOE_DEVICE=cpu \
    DOE_CPU_THREADS=1 \
    DOE_CHECKPOINT=/app/artifacts/doe-carrier-study-v1.pt \
    DOE_STATIC_DIR=/app/frontend-dist \
    PORT=10000
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 60 --retries 6 torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu
COPY backend/pyproject.toml ./
COPY backend/src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 60 --retries 6 . \
    && useradd --create-home photonica
COPY artifacts/doe-carrier-study-v1.pt ./artifacts/doe-carrier-study-v1.pt
COPY --from=frontend /app/dist ./frontend-dist
USER photonica
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn doe.api:app --host 0.0.0.0 --port \"$PORT\" --workers 1"]
