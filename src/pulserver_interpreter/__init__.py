"""Public interface."""

from importlib import metadata as _metadata

# =========
# VERSION
# =========
__version__ = _metadata.version("pulserver_interpreter")


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from . import builder  # noqa

# from . import demo  # noqa
# from . import pulseq  # noqa
