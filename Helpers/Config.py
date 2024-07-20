from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SERVICE_PORT: int
    SERVICE_HOST: str
    FFMPEG_PATH: str

    class Config:
        env_file = ".env"


def get_Settings():
    return Settings()
