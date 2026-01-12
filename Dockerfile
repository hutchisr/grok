FROM python:3.14
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/
COPY bot /app/bot

WORKDIR /app

RUN uv sync

ENTRYPOINT ["uv", "run", "--no-sync"]

CMD ["python", "-m", "bot", "-c", "/config.yaml"]
