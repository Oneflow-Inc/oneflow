import time
import logging
import os
from pathlib import Path


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[34m",  # Blue
        "INFO": "\033[92m",  # green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m",  # Red
    }

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, "\033[0m")  # Default to Reset color
        return f"{color}{log_message}\033[0m"


class ConfigurableLogger:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def configure_logging(self, name, level, log_dir=None, file_name=None):
        logger = logging.getLogger(name)

        if logger.hasHandlers():
            logger.warning("Logging handlers already exist for %s", name)
            return

        logger.setLevel(level)

        # Create a console formatter and add it to a console handler
        console_formatter = ColorFormatter(
            fmt="%(levelname)s [%(asctime)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Create a file formatter and add it to a file handler if log_dir is provided
        if log_dir:
            log_dir = Path(log_dir)
            os.makedirs(log_dir, exist_ok=True)

            file_prefix = "{}_".format(
                time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            )

            if file_name:
                log_file_name = file_prefix + file_name
            else:
                log_file_name = file_prefix + name + ".log"

            log_file = log_dir / log_file_name
            file_formatter = logging.Formatter(
                fmt="%(levelname)s [%(asctime)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        self.logger = logger


logger = ConfigurableLogger()
