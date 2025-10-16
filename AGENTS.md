# AI Agent Guide for Grok Bot

This document provides guidance for AI agents working with this Misskey/Fediverse chatbot codebase.

## Project Overview

**Grok** is a Python-based chatbot for Misskey (Fediverse) that uses DSPy and LLM endpoints to respond to mentions. The bot connects via WebSocket, processes mentions with context awareness, and can optionally use web search capabilities.

## Architecture

### Core Components

1. **`bot/bot.py`** - Main bot logic
   - WebSocket connection management
   - Message routing and mention handling
   - Note fetching with context building
   - Reply sending with mention tracking

2. **`bot/ai.py`** - AI/LLM integration
   - DSPy-based `ChatAgent` using ReAct pattern
   - Multi-endpoint fallback for reliability
   - Image description via vision models
   - Configurable tools (datetime, web search)

3. **`bot/models.py`** - Data models
   - Pydantic models for Misskey API structures
   - Configuration schema with validation
   - Note, User, and File models

4. **`bot/tools.py`** - Agent tools
   - Web search via SearXNG integration
   - Current datetime tool
   - Configurable tool functions

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
- `model_file`: Path to optimized DSPy model
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
6. Generates reply using LLM with ReAct
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
- Retries on TypeError (DSPy internal errors)
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
Add functions to [`bot/tools.py`](bot/tools.py:1):
```python
def my_tool(param: str) -> str:
    """Tool description for LLM"""
    # Implementation
    return result
```

Register in [`bot/ai.py`](bot/ai.py:42):
```python
tools = [current_datetime, my_tool]
```

### Modifying Agent Behavior
Edit the `Reply` signature in [`bot/ai.py`](bot/ai.py:16) to change input/output fields:
```python
class Reply(dspy.Signature):
    new_field: str = dspy.InputField(desc="Description")
    # ... other fields
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
| [`bot/ai.py`](bot/ai.py:1) | DSPy agent, LLM orchestration |
| [`bot/models.py`](bot/models.py:1) | Pydantic models |
| [`bot/tools.py`](bot/tools.py:1) | Agent tool functions |
| [`config.example.yaml`](config.example.yaml:1) | Configuration template |
| [`pyproject.toml`](pyproject.toml:1) | Python dependencies |
| [`Dockerfile`](Dockerfile:1) | Container image definition |
| [`mise.toml`](mise.toml:1) | Task automation |

## API Integration

### Misskey API Endpoints Used
- `POST /api/notes/create` - Send notes
- `POST /api/notes/show` - Fetch note details
- WebSocket `/streaming` - Real-time events

### DSPy Integration
- Uses DSPy 3.0+ with ReAct pattern
- Async support via `acall()` methods
- Model optimization via `user-style.ipynb` notebook
- History tracking for conversation context

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
