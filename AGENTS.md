# AI Agent Guide for Grok Bot

This document provides guidance for AI agents working with this Misskey/Fediverse chatbot codebase.

## Project Overview

**Grok** is a Python-based chatbot for Misskey (Fediverse) that uses Pydantic AI and LLM endpoints to respond to mentions. The bot connects via WebSocket, processes mentions with context awareness, and can optionally use web search capabilities.

## Architecture

### Core Components

1. **`bot/bot.py`** - Main bot logic
   - WebSocket connection management
   - Message routing and mention handling
   - Note fetching with context building
   - Reply sending with mention tracking

2. **`bot/ai.py`** - AI/LLM integration
   - Pydantic AI-based `ChatAgent` with structured output
   - Multi-endpoint fallback for reliability
   - Image description via vision models
   - Configurable tools (datetime, web search)

3. **`bot/models.py`** - Data models
   - Pydantic models for Misskey API structures
   - Configuration schema with validation
   - Note, User, and File models

4. **`bot/tools.py`** - Utility functions
   - Web search via SearXNG integration
   - Current datetime utility
   - Note: Tools are registered directly on agents in `bot/ai.py`

5. **`bot/api.py`** - HTTP client (not shown but referenced)

## Configuration

Configuration is via YAML file (see [`config.example.yaml`](config.example.yaml:1)):

### Required Fields
- `domain`: Misskey instance domain
- `url`: HTTPS API base URL
- `ws_url`: WebSocket URL
- `token`: Misskey API token
- `bot_user_id`: Bot's Misskey user ID
- `bot_username`: Bot's username (without @)
- `llm_endpoints`: List of LLM endpoints with fallback support
- `vision_endpoints`: List of vision model endpoints
- `system_prompt`: Bot personality/instructions

### Optional Fields
- `searxng_url`, `searxng_user`, `searxng_password`: Web search integration
- `max_context`: Number of parent notes to include (default: 1)
- `max_tokens`: Token limit per response
- `debug`: Enable debug logging

### LLM Endpoint Format
```yaml
llm_endpoints:
  - url: "https://api.example.com/v1"
    key: "api-key"
    model: "model-name"
    provider: "openai"  # optional, defaults to openai
```

## Key Behaviors

### Message Processing Flow
1. Bot receives mention via WebSocket
2. Ignores own mentions
3. Builds context by traversing reply chain (up to `max_context`)
4. Includes renote content if available
5. Describes any attached images using vision models
6. Generates structured reply using Pydantic AI agent
7. Filters out self-mentions from response
8. Sends reply with proper mention formatting

### Context Building
- Follows `replyId` chain backwards
- Only includes notes with text or files
- Adds renote content if present
- Maximum depth controlled by `max_context`

### Image Handling
- Extracts all images from context and current note
- Uses vision endpoints to generate descriptions
- Descriptions passed as context to main LLM

### Error Handling
- Multi-endpoint fallback for LLMs
- Automatic retry with next endpoint on failure
- WebSocket auto-reconnect on disconnect
- Task-based async processing with error logging

## Development Setup

### Prerequisites
- Python 3.13+
- `uv` package manager (REQUIRED)
- Misskey instance with API access

### Installation
```bash
# Using uv
uv sync
```

**IMPORTANT**: This project uses `uv` for dependency management and virtual environment handling. The virtual environment is located at `.venv`. When running Python commands in this repository, you MUST use one of these approaches:

1. **Use `uv run`** (recommended): `uv run python -m bot -c config.yaml`
2. **Activate the virtualenv**: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
3. **Direct venv python**: `.venv/bin/python -m bot -c config.yaml`

### Running the Bot
```bash
# Using mise task (preferred)
mise run bot

# Direct execution with uv
uv run python -m bot -c config.local.yaml

# Using the virtualenv directly
.venv/bin/python -m bot -c config.local.yaml

# Using Docker
docker build -t grok .
docker run -v /path/to/config.yaml:/config.yaml grok
```

## Code Patterns

### Adding New Tools
Register tools directly on the agent in [`bot/ai.py`](bot/ai.py) using the `@agent.tool_plain` decorator:
```python
@agent.tool_plain
def my_tool(param: str) -> str:
    """Tool description for LLM"""
    # Implementation
    return result
```

Or use `@agent.tool` if you need access to the `RunContext`:
```python
from pydantic_ai import RunContext

@agent.tool
def my_tool(ctx: RunContext[AgentDeps], param: str) -> str:
    """Tool with context access"""
    return f"User {ctx.deps.user} requested: {param}"
```

### Modifying Agent Behavior
Edit the `ReplyOutput` model in [`bot/ai.py`](bot/ai.py) to change output fields:
```python
class ReplyOutput(BaseModel):
    """Output schema for the chat agent."""
    reply: str
    """Reply to the message."""
    mentions: Optional[list[str]] = None
    """List of usernames mentioned."""
    new_field: str
    """Description of new field."""
```

Modify the `AgentDeps` model to change input dependencies:
```python
class AgentDeps(BaseModel):
    message: str
    user: str
    # Add new fields here
```

### Customizing System Prompt
Update `system_prompt` in config YAML. The prompt shapes the bot's personality and response style.

## Deployment

### Kubernetes
Deployment files in [`k8s/`](k8s/) directory:
- Uses Kustomize for configuration
- Expects config as ConfigMap or Secret
- Mise tasks for build and deploy:
  ```bash
  mise run build    # Build and push Docker image
  mise run upgrade  # Apply K8s manifests and restart
  ```

### Environment Variables
Set via config YAML. For sensitive values, use secrets management in your deployment platform.

## Testing & Debugging

### Debug Mode
Set `debug: true` in config for verbose logging.

### Manual Testing
1. Mention the bot in a Misskey note
2. Check logs for processing flow
3. Verify reply appears correctly

### Common Issues
- **No response**: Check token, user_id, WebSocket connection
- **Wrong mentions**: Verify `bot_username` matches Misskey username
- **LLM failures**: Check endpoint URLs, API keys, model names
- **Image processing fails**: Verify vision endpoints are configured

## Key Files Reference

| File | Purpose |
|------|---------|
| [`bot/bot.py`](bot/bot.py:1) | WebSocket client, message routing |
| [`bot/ai.py`](bot/ai.py:1) | Pydantic AI agent, LLM orchestration |
| [`bot/models.py`](bot/models.py:1) | Pydantic models |
| [`bot/tools.py`](bot/tools.py:1) | Utility functions |
| [`config.example.yaml`](config.example.yaml:1) | Configuration template |
| [`pyproject.toml`](pyproject.toml:1) | Python dependencies |
| [`Dockerfile`](Dockerfile:1) | Container image definition |
| [`mise.toml`](mise.toml:1) | Task automation |

## API Integration

### Misskey API Endpoints Used
- `POST /api/notes/create` - Send notes
- `POST /api/notes/show` - Fetch note details
- WebSocket `/streaming` - Real-time events

### Pydantic AI Integration
- Uses Pydantic AI with structured output
- Async support via `agent.run()` method
- OpenAI-compatible endpoints via `OpenAIChatModel`
- Multi-endpoint fallback for reliability
- Conversation history passed as context in prompts

## Performance Considerations

- Async message processing prevents blocking
- Multi-endpoint fallback for reliability
- Configurable token limits to control costs
- Image thumbnails used instead of full images
- Context depth limited by `max_context`

## Security Notes

- Store API keys securely (use secrets in production)
- Validate all external inputs via Pydantic models
- SearXNG credentials optional but recommended
- Bot ignores its own messages to prevent loops

## Contributing

When modifying this codebase:
1. Maintain async/await patterns throughout
2. Use Pydantic for all data validation
3. Add type hints to all functions
4. Update this guide for architectural changes
5. Test with multiple LLM providers
6. Consider token costs in implementations

## Troubleshooting

### Bot not responding
1. Verify WebSocket connection in logs
2. Check bot has API token with write permissions
3. Ensure `bot_user_id` matches actual user ID
4. Confirm LLM endpoints are accessible

### Context not loading
1. Check `max_context` setting
2. Verify note threading is correct in Misskey
3. Look for HTTP errors in note fetching

### LLM errors
1. Try multiple endpoint configurations
2. Check model names match provider API
3. Verify API keys are valid
4. Reduce `max_tokens` if hitting limits
