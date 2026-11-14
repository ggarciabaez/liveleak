# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.12-slim-bookworm

ENV UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    UV_CONCURRENT_DOWNLOADS=3 \
    PYTHONUNBUFFERED=1 \
    PATH="/backend/.venv/bin:$PATH" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /bin/uv
WORKDIR /backend
COPY pyproject.toml uv.lock ./
ARG DEVICE=cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --no-dev --no-emit-project --no-hashes --output-file /tmp/locked.txt \
    && sed '/^tensorrt[=-]/d' /tmp/locked.txt > /tmp/backend.txt \
    && uv venv \
    && case "$DEVICE" in \
        cpu) \
            sed -i '/^nvidia-ml-py==/! { /^\(nvidia-\|cuda-\|triton==\)/d; }' /tmp/backend.txt; \
            sed -n '/^\(torch\|torchvision\)==/s/\(==[^ ;]*\)/\1+cpu/p' /tmp/backend.txt > /tmp/torch.txt; \
            sed -i '/^\(torch\|torchvision\)==/d' /tmp/backend.txt; \
            uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch.txt \
                && rm /tmp/torch.txt ;; \
        gpu) : ;; \
        *) echo "DEVICE must be cpu or gpu" >&2; exit 1 ;; \
    esac \
    && uv pip install --no-deps -r /tmp/backend.txt \
    && uv pip check \
    && rm /tmp/locked.txt /tmp/backend.txt

COPY backend/ backend/
COPY modules/ modules/
EXPOSE 5020
# Fetch any missing model files from the GitHub Release, then start the server.
CMD ["sh", "-c", "python -m backend.fetch_models && exec python -m backend.backend"]
