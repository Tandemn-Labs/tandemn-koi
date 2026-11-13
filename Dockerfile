# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.9.26 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

RUN apt-get update \
    && apt-get install --no-install-recommends --yes tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/koi

COPY pyproject.toml uv.lock ./
COPY --from=tandemn-store pyproject.toml README.md /app/tandemn-store/
COPY --from=tandemn-store src/ /app/tandemn-store/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ ./src/

RUN groupadd --system --gid 10001 koi \
    && useradd --system --uid 10001 --gid koi --create-home --home-dir /home/koi koi \
    && install -d --owner=koi --group=koi logs/koi

ENV PATH="/app/koi/.venv/bin:$PATH"

USER koi

EXPOSE 8080

ENTRYPOINT ["tini", "--", "python", "-m", "src.service"]
CMD ["--ticks", "0", "--tick-interval-sec", "300"]
