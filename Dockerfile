FROM python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/
COPY bot /app/bot

WORKDIR /app

RUN uv sync

CMD ["uv", "run", "python", "-m", "bot", "-c", "/config.yaml"]
