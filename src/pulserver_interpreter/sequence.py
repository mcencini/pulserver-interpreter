"""Drop-in replacement for pypulseq.Sequence"""

__all__ = ["Sequence"]

import numpy as np

from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence

from .segment import create_segments as _create_segments


class Sequence:
    """
    Drop-in replacement for pypulseq.Sequence, supporting prep and eval/rt modes.
    Tracks TRID, block deduplication, amplitude/duration monitoring, and supports
    mode switching for real-time and preparation workflows.

    Attributes
    ----------
    mode_flag : str
        Current mode of the sequence ('prep', 'eval', or 'rt').
    blocks : list
        Stores all blocks added to the sequence.
    trid_array : list
        Stores TRID for each block.
    definitions : dict
        Maps TRID to lists of block indices.
    amplitudes : dict
        Maps block index to maximum amplitude.
    durations : dict
        Maps block index to minimum duration.
    """

    def __init__(self, system: Opts | None = None, use_block_cache: bool = True):
        """
        Initialize a new Sequence instance. Accepts the same arguments as pypulseq.Sequence.

        Parameters
        ----------
        system : Opts, optional
            System limits or configuration.
        use_block_cache : bool, optional
            Whether to use block cache (default: True).
        """
        self._mode = "prep"  # 'prep' or 'eval'
        self._current_trid = None  # The TRID currently being built
        self._seq = PyPulseqSequence(system=system, use_block_cache=use_block_cache)
        self._trid_events = {}
        self._tr_cursor = {}
        self._segment_cursor = {}

        self._tr_library = {}
        self._segment_library = {}
        self._block_library = PyPulseqSequence(system=system)

    def add_block(self, *args) -> None:
        """
        Add a block to the sequence, dispatching to the appropriate method based on mode.

        Parameters
        ----------
        *args : SimpleNamespace
            Events to add as a block (including possible TRID label as a SimpleNamespace).
        """
        if self._mode == "prep":
            self._add_block_prep(*args)
        elif self._mode == "eval":
            self._add_block_eval(*args)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def _add_block_prep(self, *args) -> None:
        """
        Add a block in preparation mode, tracking only TRID pattern and building the base TRs as described.

        Parameters
        ----------
        *args : SimpleNamespace
            Events to add as a block (including possible TRID label as a SimpleNamespace).
        """
        # Check if any argument is a TRID label
        trid = None
        for obj in args:
            if hasattr(obj, "label") and getattr(obj, "label", None) == "TRID":
                trid = getattr(obj, "value", None)
                break

        # If TRID > 0 and not already in definitions, start new TRID definition
        if trid is not None and trid > 0:
            if trid not in self._trid_events:
                self._trid_events[trid] = []
                self._current_trid = trid
            else:
                # If we see a TRID > 0 that is already in the definitions, stop building
                self._current_trid = None

        # Only update definition and add blocks if we are building the first instance
        if self._current_trid is not None:
            if trid is not None:
                self._trid_events[self._current_trid].append(trid)
            else:
                self._trid_events[self._current_trid].append(0)
            self._seq.add_block(*args)

    def _add_block_eval(self, *args) -> None:
        """
        In eval mode, only update variable parameters, do not add new blocks.

        Parameters
        ----------
        *args : SimpleNamespace
            Events to update (including possible TRID label as a SimpleNamespace).
        """

    @property
    def system(self):
        return self._seq.system

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["prep", "eval", "rt"]:
            raise ValueError(f"Mode (={value}) must be 'prep', 'eval', or 'rt'.")
        self._mode = value

    def create_segments(self):
        """
        Use the external segment.create_segments wrapper to perform block deduplication, segment splitting, and segment deduplication.
        Stores the resulting segment library, cursors, and block library as attributes.
        """
        seq = self._seq
        (
            self._segment_library,
            self._tr_cursor,
            self._segment_cursor,
            self._tr_library,
            self._block_library,
        ) = _create_segments(seq, self._trid_events)


# %% Local subroutines
def _unique(arr, return_index=False, return_inverse=False, return_counts=False):

    sorted_idx = np.lexsort(arr.T)
    sorted_arr = arr[sorted_idx]

    unique_mask = np.empty(arr.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = np.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
    unique_mask_idx = np.where(unique_mask)[0]

    unique_vals = sorted_arr[unique_mask_idx]

    results = [unique_vals]

    if return_index:
        index = sorted_idx[unique_mask_idx]
        results.append(index)

    if return_inverse:
        inverse = np.empty(arr.shape[0], dtype=int)
        inverse[sorted_idx] = np.cumsum(unique_mask) - 1
        results.append(inverse)

    if return_counts:
        counts = np.diff(np.append(unique_mask_idx, arr.shape[0]))
        results.append(counts)

    return tuple(results) if len(results) > 1 else results[0]


def _unique_rows_with_inverse(mat):
    # Use lexsort-based unique row finder, preserving order of first appearance
    _, idx, inv = _unique(mat, return_index=True, return_inverse=True)
    return mat[np.sort(idx)], idx[inv]
