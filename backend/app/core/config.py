from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NER-Nav API"
    app_env: str = "development"
    app_version: str = "0.1.0"

    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "ner_nav"
    postgres_user: str = "ner_nav"
    postgres_password: str

    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            if self.database_url.startswith("postgresql://"):
                return self.database_url.replace(
                    "postgresql://",
                    "postgresql+psycopg://",
                    1,
                )
            return self.database_url

        return (
            f"postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )


settings = Settings()
