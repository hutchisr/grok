from pydantic import BaseModel, Field, AnyHttpUrl, AnyUrl, WebsocketUrl, field_validator, model_validator
from typing import List, Optional, Literal
import re


class User(BaseModel):
    id: str
    name: Optional[str] = None
    username: str
    host: Optional[str] = None
    location: Optional[str] = None

    # Allow additional fields with unknown types
    class Config:
        extra = "allow"


class Note(BaseModel):
    id: str
    text: Optional[str] = None
    userId: str
    user: User
    replyId: Optional[str] = None
    renoteId: Optional[str] = None
    reply: Optional['Note'] = None
    renote: Optional['Note'] = None
    visibility: Literal["public", "home", "followers", "specified"]
    mentions: Optional[List[str]] = None

    # Allow additional fields with unknown types
    class Config:
        extra = "allow"


class MiWebsocketMessageBody(BaseModel):
    type: Optional[str] = None
    body: Optional[Note] = None
    channel: Optional[str] = None
    id: Optional[str] = None


class MiWebsocketMessage(BaseModel):
    type: str
    body: Optional[MiWebsocketMessageBody] = None


class LlmEndpoint(BaseModel):
    url: AnyHttpUrl = Field(description="endpoint url")
    key: Optional[str] = None
    model: str = Field(description="model")
    provider: Optional[str] = None


class Config(BaseModel):
    domain: str = Field(description="domain")
    url: AnyHttpUrl = Field(description="url")
    ws_url: WebsocketUrl = Field(description="ws_url")
    token: str = Field(description="token")
    channel: Optional[str] = None
    llm_endpoints: List[LlmEndpoint] = Field(description="llm_endpoints")
    max_tokens: int = Field(gt=0, description="max_tokens must be greater than 0")
    bot_user_id: str = Field(description="bot_user_id")
    bot_username: str = Field(description="bot_username")
    system_prompt: str = Field(description="system_prompt")
    system_prompt_auto: str = Field(description="system_prompt_auto")
    redis_uri: Optional[AnyUrl] = None
    redis_key_prefix: Optional[str] = Field(None, description="redis_key_prefix")
    redis_key_ttl: Optional[int] = Field(None, gt=0, description="redis_key_ttl must be greater than 0")
    max_retries: int = Field(gt=0, description="max_retries must be greater than 0")
    max_memory: int = Field(ge=0, description="max_memory must not be negative")
    searxng_url: Optional[AnyHttpUrl] = None
    searxng_user: Optional[str] = None
    searxng_password: Optional[str] = None
    model_file: Optional[str] = None
    debug: Optional[bool] = None

    @field_validator('redis_uri')
    @classmethod
    def validate_redis_uri(cls, v):
        if v is None:
            return v

        # Redis URI pattern validation
        redis_pattern = r'^(redis|rediss|redis-sentinel):\/\/(?:([^:/@\s]+)(?::([^@\s]*))?@)?([^:/@\s]+|\[[a-fA-F0-9:]+\])(?::(\d+))?(?:\/(\d+))?$'
        if not re.match(redis_pattern, str(v)):
            raise ValueError("redis_uri must be a valid Redis URI (redis:// or rediss://)")
        return v

    @model_validator(mode='after')
    def validate_redis_config(self):
        # Custom validation: if redis_uri is provided, redis_key_ttl must be positive
        if self.redis_uri and self.redis_key_ttl is not None and self.redis_key_ttl <= 0:
            raise ValueError("redis_key_ttl must be greater than 0 when redis_uri is provided")
        return self
