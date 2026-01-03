"""Storage modules for logging and evidence."""
from .logger import SystemLogger, get_logger
from .evidence import EvidenceManager

__all__ = ['SystemLogger', 'get_logger', 'EvidenceManager']
