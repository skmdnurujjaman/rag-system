from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_url: str
    openai_api_key: SecretStr
    gemini_api_key: SecretStr
    mcp_auth_token: SecretStr
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_public_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    redis_url: str = "redis://localhost:6379"

settings = Settings()
