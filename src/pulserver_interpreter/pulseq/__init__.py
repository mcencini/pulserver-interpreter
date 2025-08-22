"""
A caching wrapper for PyPulseq functions to accelerate repeated event creation. 

Supports mutable attributes, nested event structures, user-defined event functions, and amplitude-aware rescaling.

Features
--------
- Automatic caching of built-in `make_*` PyPulseq functions, excluding cheap events
  (`make_delay`, `make_label`, `make_trigger`, `make_adc`).
- Supports single events, tuples/lists, nested dicts or nested mixtures.
- Mutable attributes (`phase_offset`, `freq_offset`, `delay`, `amplitude`, 
  `freq_ppm`, `phase_ppm`) updated recursively on cached events.
- Deepcopy safety to ensure repeated sequence design passes do not interfere.
- `@cached_event` decorator for user-defined functions, with optional per-function
  additional mutable attributes.
- Safe amplitude rescaling for RF and gradient events, recomputing if amplitude
  exceeds cached maximum.
  
"""

__all__ = ["PyPulseq", "Sequence", "harmonize_gradients"]

import copy as _copy
import inspect as _inspect

from functools import wraps as _wraps

# PyPulseq decorates all make* functions automatically except the following
__EXCLUDE_CACHE__ = {"make_delay", "make_label", "make_trigger"}

# Following attributes will trigger smart scaling instead of recomputation
__DEFAULT_MUTABLE__ = {
    "amplitude",
    "delay",
    "phase_offset",
    "phase_ppm",
    "freq_offset",
    "freq_ppm",
}


class PyPulseq:
    """
    Cached wrapper for PyPulseq functions to speed up repeated event creation
    with support for mutable attributes and nested structures.

    Parameters
    ----------
    module : module
        PyPulseq module to wrap.
    mutable_attrs : set[str], optional
        Attributes considered mutable, default includes
        'phase_offset', 'freq_offset', 'delay', 'amplitude', 'freq_ppm', 'phase_ppm'.
    """

    def __init__(self, module, mutable_attrs: set[str] | None = None):
        self._module = module
        self.mutable_attrs = mutable_attrs or __DEFAULT_MUTABLE__

        # Wrap all make_* functions automatically, except cheap events
        exclude_cache = __EXCLUDE_CACHE__
        for name in dir(module):
            if name.startswith("make") and callable(getattr(module, name)):
                if name not in exclude_cache:
                    setattr(self, name, self._cached_event(getattr(module, name)))
                else:
                    setattr(self, name, getattr(module, name))
            else:
                setattr(self, name, getattr(module, name))

    # --- internal recursive mutable update ---
    def _apply_mutable_attrs(self, obj, mutable_vals: dict) -> object | None:
        """
        Recursively apply mutable attributes to obj if it has them.

        Parameters
        ----------
        obj : object
            Event object, or tuple/list/dict of events.
        mutable_vals : dict
            Dictionary of mutable attribute names and their values.

        Returns
        -------
        object | None
            Updated object, or None if recompute is needed (e.g., amplitude too large).
        """
        if isinstance(obj, dict):
            result = {}
            recompute_needed = False
            for k, v in obj.items():
                updated = self._apply_mutable_attrs(v, mutable_vals)
                if updated is None:
                    recompute_needed = True
                result[k] = updated
            if recompute_needed:
                return None
            return result

        elif isinstance(obj, (tuple, list)):
            result = []
            recompute_needed = False
            for o in obj:
                updated = self._apply_mutable_attrs(o, mutable_vals)
                if updated is None:
                    recompute_needed = True
                result.append(updated)
            if recompute_needed:
                return None
            return type(obj)(result)

        else:
            for attr, val in mutable_vals.items():
                if hasattr(obj, attr):
                    if attr == "amplitude":
                        # compute cached amplitude locally
                        if hasattr(obj, "amplitude"):
                            cached_amp = obj.amplitude
                        elif hasattr(obj, "signal"):
                            cached_amp = abs(obj.signal).max()
                        elif hasattr(obj, "waveform"):
                            cached_amp = abs(obj.waveform).max()
                        else:
                            cached_amp = val
                        if val <= cached_amp:
                            scale = val / cached_amp
                            if hasattr(obj, "signal"):
                                obj.signal *= scale
                            if hasattr(obj, "waveform"):
                                obj.waveform *= scale
                            obj.amplitude = val
                        else:
                            return None  # recompute needed
                    else:
                        setattr(obj, attr, val)
            return obj

    # --- internal caching wrapper ---
    def _cached_event(self, func, mutable_attrs: set[str] | None = None):
        """
        Internal caching wrapper.

        Parameters
        ----------
        func : callable
            Function to wrap
        mutable_attrs : set[str] | None
            Overrides or extends default mutable attributes.
        """
        combined_mutable = self.mutable_attrs.copy()
        if mutable_attrs:
            combined_mutable |= mutable_attrs

        cache = {}

        @_wraps(func)
        def wrapper(*args, **kwargs):
            # --- parse signature ---
            sig = _inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            all_args = dict(bound.arguments)

            # --- separate mutable attributes and exclude 'system' ---
            mutable_vals = {}
            key_args = {}
            for k, v in all_args.items():
                if k in combined_mutable:
                    mutable_vals[k] = v
                elif k != "system":
                    key_args[k] = v

            # --- build cache key ---
            key = (func.__name__, frozenset(key_args.items()))

            # --- check cache ---
            if key in cache:
                cached_entry = cache[key]
                event = _copy.deepcopy(cached_entry["event"])
                updated_event = self._apply_mutable_attrs(event, mutable_vals)
                if updated_event is None:
                    # recompute if amplitude too large
                    event = func(*args, **kwargs)
                    cache[key] = {"event": _copy.deepcopy(event)}
                    return _copy.deepcopy(event)
                else:
                    return updated_event

            # --- not cached: compute, store ---
            event = func(*args, **kwargs)
            cache[key] = {"event": _copy.deepcopy(event)}
            return _copy.deepcopy(event)

        wrapper.clear_cache = cache.clear
        return wrapper

    # --- decorator for user-defined event functions ---
    def cached_event(self, func=None, *, mutable_attrs: set[str] | None = None):
        """
        Decorator to cache user-defined event functions.

        Parameters
        ----------
        func : callable, optional
            Function to decorate. Can be provided as positional argument.
        mutable_attrs : set[str], optional
            Additional or overriding mutable attributes for this function.
        """

        def decorator(f):
            return self._cached_event(f, mutable_attrs=mutable_attrs)

        if func is None:
            return decorator
        else:
            return decorator(func)

    # --- clear all caches ---
    def clear_all_caches(self):
        """
        Clear all cached events for this CachedPulseq instance.
        """
        for name, fn in self.__dict__.items():
            if callable(fn) and hasattr(fn, "clear_cache"):
                fn.clear_cache()


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from .Sequence.sequence import Sequence
from .harmonize_gradients import harmonize_gradients
