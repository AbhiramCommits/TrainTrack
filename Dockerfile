FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md alembic.ini ./
COPY traintrack ./traintrack
COPY migrations ./migrations
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir .

# Copied after dependency installation so template edits do not invalidate the
# expensive torch layer.
COPY deploy ./deploy

RUN useradd --create-home --uid 10001 traintrack \
    && mkdir -p /artifacts /data \
    && chown -R traintrack:traintrack /artifacts /data
USER traintrack

EXPOSE 8010

CMD ["python", "-m", "traintrack.api"]
