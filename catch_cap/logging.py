"""Logging configuration for catch_cap."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "catch_cap",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with structured formatting.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses default structured format

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    if format_string is None:
        format_string = (
            "[%(asctime)s] %(name)s.%(levelname)s: %(message)s "
            "[%(filename)s:%(lineno)d]"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# Create default logger
logger = setup_logger()
