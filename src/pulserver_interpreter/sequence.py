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
        self._mode = "prep"  # 'prep', 'eval', or 'rt'
        self._current_trid = None  # The TRID currently being built
        self._seq = PyPulseqSequence(system=system, use_block_cache=use_block_cache)

        # --- Core libraries and mappings ---
        self.trid_events = {}  # dict: TRID -> list of TRID events (per block in TR)
        self.trid_to_block_indices = (
            {}
        )  # dict: TRID -> np.ndarray of block indices (per TR)
        self.trid_to_segment_ids = (
            {}
        )  # dict: TRID -> np.ndarray of segment IDs (per block in TR)
        self.trid_to_within_segment_idx = (
            {}
        )  # dict: TRID -> np.ndarray of within-segment indices (per block in TR)
        self.trid_definitions = {}  # dict: TRID -> tuple of segment IDs (in order)
        self.segment_library = {}  # dict: segment_id -> tuple of block IDs
        self.block_library = None  # PyPulseqSequence containing all unique blocks

        # --- State for eval mode ---
        self.eval_stats = {}  # dict: TRID -> np.ndarray (n_blocks, 8) of stats
        self.adc_count = 0
        self._eval_trid_list = []  # list of TRIDs in eval order
        self._eval_tr_block_counts = {}  # dict: TRID -> number of blocks in TR
        self._eval_tr_idx = 0  # current TR index in eval
        self._eval_block_idx = 0  # current block index in TR in eval

        # --- Counters ---
        self._prep_counts = 0  # number of add_block calls in preparation mode
        self._eval_counts = 0  # number of add_block calls in evaluation mode

    def add_block(self, *args) -> None:
        """
        Add a block to the sequence, dispatching to the appropriate method based on mode.
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
        """
        trid = None
        for obj in args:
            if hasattr(obj, "label") and getattr(obj, "label", None) == "TRID":
                trid = getattr(obj, "value", None)
                break

        # If TRID > 0 and not already in definitions, start new TRID definition
        if trid is not None and trid > 0:
            if trid not in self.trid_events:
                self.trid_events[trid] = []
                self._current_trid = trid
            else:
                self._current_trid = None

        # Only update definition and add blocks if we are building the first instance
        if self._current_trid is not None:
            if trid is not None:
                self.trid_events[self._current_trid].append(trid)
            else:
                self.trid_events[self._current_trid].append(0)
            self._seq.add_block(*args)

        self._prep_counts += 1

    def _add_block_eval(self, *args) -> None:
        """
        In eval mode, perform a dry run to determine, for each block within TR, the maximum unsigned values of rf amplitudes and gradient amplitudes on the three axes among all TR instances.
        Uses arrays created during prep step. Raises error if prep was not run.
        Also counts the number of ADC events encountered during the dry run.
        Optimized: directly inspects event objects, no sequence/block creation.
        """
        if not self.trid_events or not self.trid_to_segment_ids:
            raise RuntimeError(
                "Eval mode requires a prior prep step with valid TRID events and segment mappings."
            )

        # First time, preallocate
        if self._eval_counts == 0:
            self._eval_trid_list = list(self.trid_events.keys())
            self._eval_tr_block_counts = {
                trid: len(self.trid_events[trid]) for trid in self.trid_events
            }
            for trid, events in self.trid_events.items():
                arr = np.zeros((len(events), 8), dtype=float)
                arr[:, 2] = 1  # default gx sign +
                arr[:, 4] = 1  # default gy sign +
                arr[:, 6] = 1  # default gz sign +
                arr[:, 0] = np.inf  # duration: start with inf, will take min
                self.eval_stats[trid] = arr

        curr_trid = self._eval_trid_list[self._eval_tr_idx]
        trsize = self._eval_tr_block_counts[curr_trid]
        arr = self.eval_stats[curr_trid]
        idx = self._eval_block_idx

        rf_amp = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        duration = 0.0

        for obj in args:
            if isinstance(obj, float):
                duration = max(duration, obj)
                continue
            typ = getattr(obj, "type", None)
            if typ == "rf":
                rf_amp = np.max(np.abs(obj.signal))
                duration = max(duration, obj.delay + obj.shape_dur)
            elif typ == "trap":
                if obj.channel == "x":
                    gx = obj.amplitude
                elif obj.channel == "y":
                    gy = obj.amplitude
                elif obj.channel == "z":
                    gz = obj.amplitude
                duration = max(
                    duration,
                    obj.delay
                    + obj.rise_time
                    + obj.flat_time
                    + obj.fall_time
                    + obj.fall_time,
                )
            elif typ == "grad":
                maxval = np.argmax(np.abs(obj.waveform))
                if obj.channel == "x":
                    gx = obj.waveform[maxval]
                elif obj.channel == "y":
                    gy = obj.waveform[maxval]
                elif obj.channel == "z":
                    gz = obj.waveform[maxval]
                duration = max(duration, obj.delay + obj.shape_dur)
            elif typ == "adc":
                self.adc_count += 1
                if hasattr(obj, "duration"):
                    duration = max(duration, obj.delay + obj.duration)
                else:
                    duration = max(duration, obj.delay + obj.num_samples * obj.dwell)
            elif typ == "delay":
                duration = max(duration, obj.delay)

        arr[idx, 0] = min(arr[idx, 0], duration)
        arr[idx, 1] = max(arr[idx, 1], abs(rf_amp))
        if self._eval_tr_idx == 0 and gx != 0:
            arr[idx, 2] = np.sign(gx)
        arr[idx, 3] = max(arr[idx, 3], abs(gx))
        if self._eval_tr_idx == 0 and gy != 0:
            arr[idx, 4] = np.sign(gy)
        arr[idx, 5] = max(arr[idx, 5], abs(gy))
        if self._eval_tr_idx == 0 and gz != 0:
            arr[idx, 6] = np.sign(gz)
        arr[idx, 7] = max(arr[idx, 7], abs(gz))

        self._eval_block_idx += 1
        if self._eval_block_idx >= trsize:
            self._eval_block_idx = 0
            self._eval_tr_idx += 1
            if self._eval_tr_idx >= len(self._eval_trid_list):
                self._eval_tr_idx = 0

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
        Stores the resulting segment library, mapping arrays, and block library as attributes.
        """
        (
            self.segment_library,
            self.trid_to_segment_ids,
            self.trid_to_within_segment_idx,
            self.trid_definitions,
            self.block_library,
        ) = _create_segments(self._seq, self.trid_events)


# %% utils
def _get_block(*args):
    _seq = PyPulseqSequence(system=Opts(max_grad=np.inf, max_slew=np.inf))
    _seq.add_block(*args)
    return _seq.get_block(1)
