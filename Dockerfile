FROM python:3.14
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN uv sync --no-install-project

COPY bot /app/bot

RUN uv sync

ENTRYPOINT ["uv", "run", "--no-sync"]

CMD ["python", "-m", "bot", "-c", "/config.yaml"]
