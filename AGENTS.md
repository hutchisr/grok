# Grok Bot

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, and optional Redis-backed social credit system.

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# Lint & format
uv run ruff check bot/
uv run ruff format --check bot/

# Docker
docker build -t grok . && docker run -v /path/to/config.yaml:/config.yaml grok

# Kubernetes
mise run build      # Build and push Docker image
mise run upgrade    # Apply K8s manifests and restart
```

**Important:** Always use `uv run` or `.venv/bin/python` — never bare `python`.

## Architecture

| File | Purpose |
|------|---------|
| `bot/bot.py` | WebSocket client, mention handling, context building, reply sending |
| `bot/ai.py` | `ChatAgent` class — Pydantic AI agent with `FallbackModel`, vision support |
| `bot/models.py` | Pydantic models: `Config`, `Note`, `User`, `MiFile`, WS message types |
| `bot/tools.py` | `build_tools()` factory — datetime, web search, create_note, search_users/notes, social credit tools |
| `bot/api.py` | HTTP client utilities |
| `bot/cli.py` | CLI entry point and argument parsing |

## Config Schema (`config.yaml`)

Required fields:
- `domain`, `url` (HTTPS), `ws_url` (WebSocket), `token`
- `bot_user_id`, `bot_username`
- `llm_models`: list of model strings (e.g. `"openrouter:anthropic/claude-3.5-sonnet"`)
- `system_prompt`, `max_tokens`, `max_retries`

Optional fields:
- `vision`: bool (default `true`) — pass images directly to the main LLM
- `vision_models`: legacy, unused when `vision=true`
- `system_prompt_auto` + `auto_post_interval`: autonomous posting (interval in seconds)
- `searxng_url`, `searxng_user`, `searxng_password`: web search via SearXNG
- `redis_url`, `redis_password`, `redis_db`: Redis for social credit system
- `max_context`: parent notes to include (default 1)
- `http_timeout_seconds`: HTTP timeout (default 30.0)
- `channel`, `debug`

## Key Patterns

### Agent setup (`bot/ai.py`)
- `AgentDeps` is a **dataclass** (not BaseModel) with `username`, `social_credit_score`, `adjusted_credit_users`
- Agent uses `output_type=str` (plain string output, not structured)
- Tools are built via `build_tools()` in `bot/tools.py` and passed to `Agent(..., tools=tools)`
- `FallbackModel` wraps multiple `llm_models` for automatic failover
- Social credit score is injected via a dynamic system prompt function

### Adding tools
Add tools inside `build_tools()` in `bot/tools.py` as plain functions or async functions, then append to the `tools` list:
```python
def my_tool(param: str) -> str:
    """Tool description for LLM."""
    return result

tools.append(my_tool)
```
For tools needing `RunContext`, use the signature `async def my_tool(ctx: RunContext[object], ...) -> str:`.

### Message flow
1. WebSocket mention received → `Bot` ignores own mentions
2. Reply chain traversed (up to `max_context`) to build `message_history`
3. Images passed inline via `ImageUrl` when `vision=true`
4. `ChatAgent.run()` calls Pydantic AI agent with fallback model
5. Reply sent via Misskey API with proper mention formatting

### Autonomous posting
When `system_prompt_auto` and `auto_post_interval` are configured, `ChatAgent.run_auto()` generates unprompted timeline posts on a timer.

## Available Tools (runtime)
- `current_datetime_tool` — always available
- `search_web` — when `searxng_url` configured
- `create_note` — create Misskey posts with visibility/mention control
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit`, `get_social_credit_history`, `get_social_credit_leaderboard`

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Fix any issues before considering the task complete
