"""
Configuration management for PV-CSER Pro application.

Handles loading and managing application configuration from YAML files
and environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5432
    database: str = "pv_cser_pro"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    # Application settings
    app_name: str = "PV-CSER Pro"
    app_version: str = "1.0.0"
    debug_mode: bool = False

    # Data settings
    max_upload_size_mb: int = 200
    supported_file_types: list = field(
        default_factory=lambda: [".csv", ".xlsx", ".xls"]
    )

    # Calculation settings
    default_irradiance_stc: float = 1000.0  # W/m²
    default_temperature_stc: float = 25.0   # °C
    default_am_stc: float = 1.5             # Air mass

    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    config_dir: Path = field(default_factory=lambda: Path("config"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AppConfig":
        """Create configuration from dictionary."""
        db_config = DatabaseConfig(**config_dict.get("database", {}))

        return cls(
            app_name=config_dict.get("app_name", cls.app_name),
            app_version=config_dict.get("app_version", cls.app_version),
            debug_mode=config_dict.get("debug_mode", cls.debug_mode),
            max_upload_size_mb=config_dict.get("max_upload_size_mb", cls.max_upload_size_mb),
            supported_file_types=config_dict.get(
                "supported_file_types", [".csv", ".xlsx", ".xls"]
            ),
            default_irradiance_stc=config_dict.get("default_irradiance_stc", 1000.0),
            default_temperature_stc=config_dict.get("default_temperature_stc", 25.0),
            default_am_stc=config_dict.get("default_am_stc", 1.5),
            database=db_config,
            data_dir=Path(config_dict.get("data_dir", "data")),
            config_dir=Path(config_dict.get("config_dir", "config")),
        )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from YAML file and environment variables.

    Args:
        config_path: Optional path to configuration file.
                    Defaults to config/app_config.yaml

    Returns:
        AppConfig instance with loaded configuration
    """
    config_dict: Dict[str, Any] = {}

    # Try to load from YAML file
    if config_path is None:
        config_path = "config/app_config.yaml"

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f) or {}

    # Override with environment variables
    env_mappings = {
        "PV_CSER_APP_NAME": "app_name",
        "PV_CSER_DEBUG": "debug_mode",
        "PV_CSER_MAX_UPLOAD_MB": "max_upload_size_mb",
    }

    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            if config_key == "debug_mode":
                config_dict[config_key] = value.lower() in ("true", "1", "yes")
            elif config_key == "max_upload_size_mb":
                config_dict[config_key] = int(value)
            else:
                config_dict[config_key] = value

    # Database configuration from environment
    db_config = config_dict.get("database", {})
    db_env_mappings = {
        "DATABASE_URL": None,  # Full connection string
        "DB_HOST": "host",
        "DB_PORT": "port",
        "DB_NAME": "database",
        "DB_USER": "username",
        "DB_PASSWORD": "password",
    }

    for env_var, config_key in db_env_mappings.items():
        if env_var in os.environ:
            if env_var == "DATABASE_URL":
                # Parse full connection string if provided
                db_config["connection_url"] = os.environ[env_var]
            elif config_key == "port":
                db_config[config_key] = int(os.environ[env_var])
            else:
                db_config[config_key] = os.environ[env_var]

    config_dict["database"] = db_config

    return AppConfig.from_dict(config_dict)


# IEC 61853 Standard Constants
IEC_CONSTANTS = {
    "STC_IRRADIANCE": 1000.0,      # W/m²
    "STC_TEMPERATURE": 25.0,        # °C
    "STC_AIR_MASS": 1.5,
    "NOCT_IRRADIANCE": 800.0,       # W/m²
    "NOCT_AMBIENT_TEMP": 20.0,      # °C
    "NOCT_WIND_SPEED": 1.0,         # m/s
    "STEFAN_BOLTZMANN": 5.67e-8,    # W/(m²·K⁴)
}

# Standard irradiance levels for power matrix (IEC 61853-1)
STANDARD_IRRADIANCE_LEVELS = [100, 200, 400, 600, 800, 1000, 1100]  # W/m²

# Standard temperature levels for power matrix (IEC 61853-1)
STANDARD_TEMPERATURE_LEVELS = [15, 25, 50, 75]  # °C
