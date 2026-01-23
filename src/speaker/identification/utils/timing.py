"""
utils/timing.py

Helpers for measuring and logging elapsed time.
"""

import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def now() -> float:
    """Return current timestamp in seconds."""
    return time.time()


def elapsed(t0: float) -> float:
    """Return seconds elapsed since t0."""
    return time.time() - t0


@contextmanager
def time_block(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to log elapsed time for a named block.
    Usage:
        with time_block("step name"):
            do_work()
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    t0 = now()
    try:
        yield
    finally:
        logger.info("%s took %.2f s", name, elapsed(t0))


def timeit(fn: Callable):
    """
    Decorator to log execution time of a function.
    """
    def wrapper(*args, **kwargs):
        t0 = now()
        result = fn(*args, **kwargs)
        logger.info("%s took %.2f s", fn.__name__, elapsed(t0))
        return result
    return wrapper
