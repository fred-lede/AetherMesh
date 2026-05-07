"""
Structured logging configuration for AI Inference Hub.

Provides JSON-formatted logs with correlation IDs for distributed tracing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def __init__(self, include_extra: bool = True) -> None:
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: dict[str, Any] = {
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
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            extra = {
                "process_id": record.process,
                "thread_id": record.thread,
                "thread_name": record.threadName,
            }
            # Add correlation ID if present
            if hasattr(record, "correlation_id"):
                extra["correlation_id"] = record.correlation_id
            if hasattr(record, "request_id"):
                extra["request_id"] = record.request_id
            if hasattr(record, "user_id"):
                extra["user_id"] = record.user_id
            log_entry["extra"] = extra

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to logs."""

    def __init__(self, correlation_id: str | None = None) -> None:
        super().__init__()
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]

    def filter(self, record: logging.LogRecord) -> logging.LogRecord:
        if not hasattr(record, "correlation_id"):
            record.correlation_id = self.correlation_id
        return record


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for AI Inference Hub.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True) or colored (False)
        log_file: Optional file path for file logging
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydantic").setLevel(logging.WARNING)


def get_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())[:8]


# Auto-configure on import if AIIH_LOG_JSON is set
if os.getenv("AIIH_LOG_JSON", "false").lower() == "true":
    configure_logging(json_format=True)
elif os.getenv("AIIH_LOG_LEVEL"):
    configure_logging(level=os.getenv("AIIH_LOG_LEVEL", "INFO"), json_format=False)