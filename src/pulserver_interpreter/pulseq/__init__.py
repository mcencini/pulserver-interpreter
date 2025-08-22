"""Drop-in replacement PyPulseq utils."""

__all__ = ["PyPulseq", "Sequence", "harmonize_gradients"]

from functools import wraps

import inspect
import copy


class PyPulseq:
    def __init__(self, pp):
        self._pp = pp

    def __getattr__(self, name):
        # Pass through all attributes/methods not explicitly overridden
        return getattr(self._pp, name)

    def cached_event(self, func=None):
        """
        Decorator to cache PyPulseq event creation functions.

        Cache key is based only on the actually provided arguments.
        """
        if func is None:
            return self.cached_event

        cache = {}
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Construct cache key using only provided arguments
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            key = tuple(sorted(bound.arguments.items()))

            if key in cache:
                cached_value = cache[key]
                return copy.deepcopy(cached_value)

            # Not cached yet: compute and store
            event = func(*args, **kwargs)
            cache[key] = copy.deepcopy(event)
            return event

        # Expose cache clearing method
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from .Sequence.sequence import Sequence
from .harmonize_gradients import harmonize_gradients
