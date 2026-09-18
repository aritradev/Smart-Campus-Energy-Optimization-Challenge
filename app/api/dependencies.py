"""HTTP API dependencies."""
from app.config import Settings, get_settings


__all__ = ["Settings", "get_settings", "settings_dep"]


def settings_dep() -> Settings:
    return get_settings()
