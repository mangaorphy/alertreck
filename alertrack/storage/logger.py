"""
Persistent Logging Module
==========================
Handles system logging to file with rotation and log levels.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config import LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOGS_DIR


class SystemLogger:
    """
    Centralized logging for ALERTRACK system.
    Logs to both file and console with rotation.
    """
    
    def __init__(self, name: str = "ALERTRACK"):
        """
        Initialize system logger.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with file and console handlers."""
        # Set log level
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized: {self.name}, level={LOG_LEVEL}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback."""
        self.logger.exception(message)


# Global logger instance
_global_logger = None


def get_logger(name: str = "ALERTRACK") -> SystemLogger:
    """
    Get global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        SystemLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = SystemLogger(name)
    
    return _global_logger


if __name__ == "__main__":
    print("\n📝 Testing SystemLogger...")
    print("=" * 60)
    
    # Create logger
    logger = get_logger()
    
    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # Test exception logging
    try:
        1 / 0
    except Exception:
        logger.exception("Caught an exception")
    
    print(f"\nLog file: {LOG_FILE}")
    print(f"Log level: {LOG_LEVEL}")
    print(f"Max size: {LOG_MAX_BYTES / 1024 / 1024:.1f} MB")
    print(f"Backup count: {LOG_BACKUP_COUNT}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete! Check log file for output.")
