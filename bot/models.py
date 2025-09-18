from pydantic import BaseModel, Field, AnyHttpUrl, WebsocketUrl
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
    reply: Optional['Note'] = None
    renote: Optional['Note'] = None
    visibility: Literal["public", "home", "followers", "specified"]
    mentions: Optional[List[str]] = None
    files: Optional[List[MiFile]]

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
    vision_endpoints: List[LlmEndpoint] = Field()
    max_tokens: int = Field(gt=0)
    bot_user_id: str = Field(description="bot_user_id")
    bot_username: str = Field(description="bot_username")
    system_prompt: str = Field(description="system_prompt")
    system_prompt_auto: str = Field(description="system_prompt_auto")
    max_retries: int = Field(gt=0)
    searxng_url: Optional[AnyHttpUrl] = None
    searxng_user: Optional[str] = None
    searxng_password: Optional[str] = None
    model_file: Optional[str] = None
    max_context: int = Field(gt=0, default=1, description="Number of context messages to include")
    debug: Optional[bool] = None
