"""Drop-in replacement PyPulseq utils."""

__all__ = ["cached_event", "PyPulseq", "Sequence", "harmonize_gradients"]

import inspect
import copy

from functools import wraps
from types import SimpleNamespace

from pypulseq import calc_duration
from pypulseq import Opts

__CACHED_FUN__ = [
    "make_block_pulse",
    "make_gauss_pulse",
    "make_sinc_pulse",
    "make_sigpy_pulse",
    "make_trapezoid",
]


def _make_hashable(obj):
    """
    Recursively convert common mutable types to immutable/hashable equivalents.
    Excludes Opts instances from hash.
    """
    if isinstance(obj, Opts):
        return "<Opts>"  # fixed placeholder, all Opts are considered identical
    elif isinstance(obj, (int, float, str, type(None), bool)):
        return obj
    elif isinstance(obj, (tuple, list)):
        return tuple(_make_hashable(o) for o in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, SimpleNamespace):
        return _make_hashable(vars(obj))
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(o) for o in obj))
    elif hasattr(obj, "__iter__"):
        return tuple(_make_hashable(o) for o in obj)
    else:
        return repr(obj)


def cached_event(func):
    """
    Decorator to cache Pulseq event creation functions.
    Each event returned will store its precomputed duration as `_duration`.
    Works with functions returning SimpleNamespace or list/tuple of SimpleNamespace.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__, _make_hashable(args), _make_hashable(kwargs))
        if key in cache:
            return copy.deepcopy(cache[key])

        out = func(*args, **kwargs)

        if isinstance(out, SimpleNamespace):
            events = [out]
        elif isinstance(out, (list, tuple)) and all(
            isinstance(e, SimpleNamespace) for e in out
        ):
            events = list(out)
        else:
            raise TypeError(
                "Cached function must return SimpleNamespace or list/tuple thereof."
            )

        for e in events:
            e._duration = calc_duration(e)

        result = events if len(events) > 1 else events[0]
        cache[key] = copy.deepcopy(result)

        return result

    return wrapper


class PyPulseq:
    """
    Wrapper around pypulseq that adds per-function caching
    for all `make_*` functions, plus a decorator for custom events.
    """

    def __init__(self, pp):
        self._pp = pp

        # auto-decorate all make_* functions
        for name, fn in inspect.getmembers(pp, inspect.isfunction):
            if name in __CACHED_FUN__:
                setattr(self, name, cached_event(fn))

        # expose the decorator as a method
        self.cached_event = cached_event

    def __getattr__(self, name):
        """
        Forward all other attributes and methods from wrapped pypulseq.
        """
        return getattr(self._pp, name)


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from .Sequence.sequence import Sequence
from .harmonize_gradients import harmonize_gradients
