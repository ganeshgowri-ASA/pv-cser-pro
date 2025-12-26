"""
Logging Configuration for PV-CSER Pro.

Provides centralized logging configuration with support for:
- Console and file logging
- Log rotation
- JSON and text formats
- Environment-based configuration
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_log_level(level_str: str) -> int:
    """Convert log level string to logging constant."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str.upper(), logging.INFO)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_rotation: bool = True,
    max_size_mb: int = 10,
    backup_count: int = 5,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Set up application-wide logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses LOG_FILE env var or logs to console only)
        log_format: Log format type ('json' or 'text')
        enable_rotation: Enable log file rotation
        max_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
        enable_console: Enable console logging

    Returns:
        Root logger instance
    """
    # Get configuration from environment or parameters
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_file = log_file or os.getenv("LOG_FILE")
    log_format = log_format or os.getenv("LOG_FORMAT", "text")
    enable_rotation = str(os.getenv("LOG_ROTATION", str(enable_rotation))).lower() == "true"
    max_size_mb = int(os.getenv("LOG_MAX_SIZE_MB", max_size_mb))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", backup_count))

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(get_log_level(log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    if log_format.lower() == "json":
        file_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Use colored formatter for console if running in terminal
        if sys.stdout.isatty():
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            console_formatter = file_formatter

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(get_log_level(log_level))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")

        file_handler.setLevel(get_log_level(log_level))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set levels for third-party loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding context to log messages.

    Usage:
        logger = LoggerAdapter(get_logger(__name__), {"session_id": "abc123"})
        logger.info("Processing module")  # Will include session_id in output
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra context to log messages."""
        extra = kwargs.get("extra", {})
        extra["extra_fields"] = self.extra
        kwargs["extra"] = extra
        return msg, kwargs


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and return values.

    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.

    Usage:
        @log_execution_time(logger)
        def my_function():
            pass
    """
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        return wrapper
    return decorator


# Initialize logging on module import with default settings
_root_logger = None


def init_logging() -> logging.Logger:
    """Initialize logging with environment-based configuration."""
    global _root_logger
    if _root_logger is None:
        _root_logger = setup_logging()
    return _root_logger


# Pre-configured loggers for different components
def get_db_logger() -> logging.Logger:
    """Get logger for database operations."""
    return logging.getLogger("pv_cser.database")


def get_calc_logger() -> logging.Logger:
    """Get logger for calculation operations."""
    return logging.getLogger("pv_cser.calculations")


def get_api_logger() -> logging.Logger:
    """Get logger for API operations."""
    return logging.getLogger("pv_cser.api")


def get_export_logger() -> logging.Logger:
    """Get logger for export operations."""
    return logging.getLogger("pv_cser.exports")


def get_validation_logger() -> logging.Logger:
    """Get logger for validation operations."""
    return logging.getLogger("pv_cser.validation")
