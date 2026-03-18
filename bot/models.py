from pydantic import BaseModel, Field, AnyHttpUrl, WebsocketUrl, model_validator
from typing import List, Optional, Literal


class User(BaseModel):
    id: str
    name: Optional[str] = None
    username: str
    host: Optional[str] = None
    location: Optional[str] = None

    # Allow additional fields with unknown types
    class Config:
        extra = "allow"


class MiFile(BaseModel):
    id: str
    type: str
    thumbnailUrl: Optional[str] = None
    url: Optional[str] = None

    class Config:
        extra = "allow"


class Note(BaseModel):
    id: str
    text: Optional[str] = None
    userId: str
    user: User
    replyId: Optional[str] = None
    renoteId: Optional[str] = None
    reply: Optional["Note"] = None
    renote: Optional["Note"] = None
    visibility: Literal["public", "home", "followers", "specified"]
    mentions: Optional[List[str]] = None
    files: Optional[List[MiFile]]

    # Allow additional fields with unknown types
    class Config:
        extra = "allow"


class MiChannelConnectParams(BaseModel):
    withRenotes: bool = True


class MiChannelConnectBody(BaseModel):
    channel: str
    id: str
    params: Optional[MiChannelConnectParams] = None


class MiChannelConnect(BaseModel):
    type: Literal["connect"] = "connect"
    body: MiChannelConnectBody


class MiWebsocketMessageBody(BaseModel):
    type: Optional[str] = None  # usually `mention` or `note`
    body: Optional[Note] = None
    channel: Optional[str] = None
    id: Optional[str] = None


class MiWebsocketMessage(BaseModel):
    type: str
    body: Optional[MiWebsocketMessageBody] = None


class Config(BaseModel):
    domain: str = Field(description="domain")
    url: AnyHttpUrl = Field(description="url")
    ws_url: WebsocketUrl = Field(description="ws_url")
    token: str = Field(description="token")
    channel: Optional[str] = None
    llm_models: List[str] = Field(description="LLM model strings (e.g., 'openrouter:anthropic/claude-3.5-sonnet')")
    vision: bool = Field(default=True, description="Enable vision (pass images directly to the main LLM)")
    vision_models: Optional[List[str]] = Field(
        default=None, description="Vision model strings (legacy, unused when vision=True)"
    )
    max_tokens: int = Field(gt=0)
    bot_user_id: str = Field(description="bot_user_id")
    bot_username: str = Field(description="bot_username")
    system_prompt: str = Field(description="system_prompt")
    system_prompt_auto: Optional[str] = Field(
        default=None,
        description="System prompt for autonomous (unprompted) posts",
    )
    auto_post_interval: Optional[int] = Field(
        default=None,
        gt=0,
        description="Interval in seconds between autonomous posts (None = disabled)",
    )
    auto_post_jitter: int = Field(
        default=0,
        ge=0,
        description="Random jitter in seconds added/subtracted from auto_post_interval",
    )
    auto_reply_enabled: bool = Field(
        default=False,
        description="Enable automatic replies to timeline notes",
    )
    auto_reply_interval: int = Field(
        default=900,
        gt=0,
        description="Minimum seconds between automatic replies",
    )
    auto_reply_jitter: int = Field(
        default=0,
        ge=0,
        description="Random jitter in seconds added/subtracted from auto_reply_interval",
    )
    max_retries: int = Field(gt=0)

    @model_validator(mode="after")
    def check_auto_post_config(self) -> "Config":
        if self.auto_post_interval and not self.system_prompt_auto:
            raise ValueError("system_prompt_auto is required when auto_post_interval is set")
        return self

    http_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="HTTP client timeout in seconds",
    )
    searxng_url: Optional[AnyHttpUrl] = None
    searxng_user: Optional[str] = None
    searxng_password: Optional[str] = None
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL (redis://host:port/db)")
    redis_password: Optional[str] = Field(default=None, description="Redis password for authentication")
    redis_db: Optional[int] = Field(default=0, ge=0, description="Redis database number (0-15)")
    max_context: int = Field(gt=0, default=1, description="Number of context messages to include")
    debug: Optional[bool] = None
