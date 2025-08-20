"""Drop-in replacement for pypulseq.Sequence with TR/segment tracking"""

__all__ = ["Sequence"]

import datetime

import numpy as np

from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence

from pypulseq.block_to_events import block_to_events as _block_to_events

from .harmonize_grad import harmonize_gradients as __harmonize_gradients__
from .segment import get_seq_structure as _get_seq_structure


class Sequence:
    """
    Drop-in replacement for pypulseq.Sequence, supporting prep mode, TR/segment tracking.

    Attributes
    ----------
    mode : str
        Current mode of the sequence ('prep', 'eval', or 'rt').
    _seq : PyPulseqSequence
        Internal PyPulseqSequence instance.
    first_tr_instances_trid_labels : dict
        Dictionary of first TR instance labels for each TRID.
    unique_blocks : dict
        Dictionary of unique block IDs to block objects.
    segments : dict
        Dictionary of segment IDs to tuples of block IDs.
    trs : dict
        Dictionary of TR IDs to tuples of segment IDs.
    block_trid : np.ndarray
        TR ID for each block in the sequence.
    block_within_tr : np.ndarray
        Within-TR index (0..len(TR)-1) for each block.
    block_segment_id : np.ndarray
        Segment ID for each block (filled during build_segments).
    block_within_segment : np.ndarray
        Within-segment index for each block (filled during build_segments).
    block_id : np.ndarray
        Block ID for each block (matching keys in ``unique_blocks``).

    """

    def __init__(self, system: Opts | None = None, use_block_cache: bool = True):
        """
        Initialize a new Sequence instance.

        Parameters
        ----------
        system : Opts, optional
            System limits or configuration.
        use_block_cache : bool, optional
            Whether to use block cache (default: True).
        """
        self._system = system
        self._use_block_cache = use_block_cache
        self.clear()

    def clear(self):
        """
        Reset internal structure.
        """
        self._mode = "prep"  # 'prep', 'eval', or 'rt'
        self._seq = PyPulseqSequence(
            system=self._system, use_block_cache=self._use_block_cache
        )

        # --- TR/block tracking ---
        self.first_tr_instances_trid_labels = (
            {}
        )  # dict: TRID -> first TR instance block labels
        self._current_trid = None
        self._current_block = 0
        self._within_tr = 0  # position within current TR

        # --- Global arrays ---
        self.block_tr_starts = []
        self.block_trid = []
        self.block_within_tr = []
        self.block_segment_id = None
        self.block_within_segment = None
        self.block_id = None

        # --- Segment/Block libraries ---
        self.unique_blocks = None
        self.segments = None
        self.trs = None

        # --- Status flags ---
        self.prepped = False
        self.evaluated = False

        # --- Initial parameters ---
        self.initial_tr_status = {}

        # --- Sequence info ----
        self.adc_count = 0
        self._n_total_segments = 0
        self._total_duration = 0.0

        # --- Real Time helpers ---

    def clear_buffer(self):
        self._seq = PyPulseqSequence(
            system=self._system, use_block_cache=self._use_block_cache
        )
        self._current_block = 0
        self._start_block = 0
        self._end_block = np.inf

    def add_block(self, *args) -> None:
        """Add a block to the sequence, dispatching to the appropriate method based on mode."""
        if self._mode == "prep":
            self._add_block_prep(*args)
        elif self._mode == "eval":
            self._add_block_eval(*args)
        elif self._mode == "rt":
            self._add_block_rt(*args)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

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

    @property
    def total_duration(self):
        return f"{datetime.timedelta(seconds=self._total_duration)}".split(".")[0]

    def get_seq_structure(self):
        """
        Use the external segment.build_segments wrapper to perform block deduplication, segment splitting, and segment deduplication.
        Stores the resulting segment library, mapping arrays, and block library as attributes.
        """
        if self.prepped:
            raise ValueError(
                "Sequence is already prepared. Please call clear() before parsing structure again."
            )
        (
            self.trs,
            self.segments,
            self.unique_blocks,
            self.block_trid,
            self.block_within_tr,
            self.block_segment_id,
            self.block_within_segment,
            self.block_id,
            self._n_total_segments,
        ) = _get_seq_structure(
            self._seq, self.first_tr_instances_trid_labels, self.block_trid
        )

        # Preallocate initial TR status
        self.initial_tr_status = {
            k: np.zeros((v.blocks.size, 5), dtype=float) for k, v in self.trs.items()
        }  # (dur, rf, gx, gy, gz)
        for k in self.initial_tr_status:
            self.initial_tr_status[k][:, 0] = np.inf
        self.gradient_signs = {
            k: np.zeros((v.blocks.size, 3), dtype=int) for k, v in self.trs.items()
        }  # (x, y, z)

        self.prepped = True

    def get_initial_tr_status(self):
        if not self.prepped:
            raise RuntimeError("Eval mode requires a valid sequence structure.")

        if self.evaluated:
            raise ValueError(
                "Sequence is already evaluated. Please call clear() before parsing initial TR status again."
            )
        for k in self.initial_tr_status:
            self.initial_tr_status[k][:, 2] *= self.gradient_signs[k][:, 0]
            self.initial_tr_status[k][:, 3] *= self.gradient_signs[k][:, 1]
            self.initial_tr_status[k][:, 4] *= self.gradient_signs[k][:, 2]
        self.gradient_signs = None
        self.evaluated = True

    def _add_block_prep(self, *args) -> None:
        """Add a block in preparation mode, tracking TRID, within-TR index, and first TR instance labels."""
        if self.prepped:
            raise ValueError(
                "Sequence is already prepared. Please call clear() running another preparation pass."
            )

        trid_label = 0
        for obj in args:
            if hasattr(obj, "label") and getattr(obj, "label", None) == "TRID":
                trid_label = getattr(obj, "value", 0)
                break

        # If TRID > 0 and not already in definitions, start new TRID definition
        if trid_label > 0:
            self._within_tr = 0
            if trid_label not in self.first_tr_instances_trid_labels:
                self.first_tr_instances_trid_labels[trid_label] = []
                self._current_trid = trid_label
            else:
                self._current_trid = None

        # Only update definition and add blocks if we are building the first instance
        if self._current_trid is not None:
            self.first_tr_instances_trid_labels[self._current_trid].append(trid_label)
            self._seq.add_block(*args)

        # Update global arrays for every block
        self.block_trid.append(trid_label)
        self.block_within_tr.append(self._within_tr)
        self._within_tr += 1

    def _add_block_eval(self, *args) -> None:
        """Add a block in evaluation mode, keeping max rf and gradient amplitudes and minimum duration."""
        if not self.prepped:
            raise RuntimeError("Eval mode requires a valid sequence structure.")

        if self.evaluated:
            raise ValueError(
                "Sequence is already evaluated. Please call clear() running another evaluation pass."
            )

        # Default values
        duration = 0.0
        rf_amp = 0.0
        gx_amp = 0.0
        gy_amp = 0.0
        gz_amp = 0.0
        gx_sign = 1
        gy_sign = 1
        gz_sign = 1

        # parse duration, rf_amp, grad amplitudes and sign
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
                    gx_amp = np.abs(obj.amplitude)
                    gx_sign = np.sign(obj.amplitude)
                elif obj.channel == "y":
                    gy_amp = np.abs(obj.amplitude)
                    gy_sign = np.sign(obj.amplitude)
                elif obj.channel == "z":
                    gz_amp = np.abs(obj.amplitude)
                    gz_sign = np.sign(obj.amplitude)
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
                    gx_amp = np.abs(obj.waveform[maxval])
                    gx_sign = np.sign(obj.waveform[maxval])
                elif obj.channel == "y":
                    gy_amp = np.abs(obj.waveform[maxval])
                    gy_sign = np.sign(obj.waveform[maxval])
                elif obj.channel == "z":
                    gz_amp = np.abs(obj.waveform[maxval])
                    gz_sign = np.sign(obj.waveform[maxval])
                duration = max(duration, obj.delay + obj.shape_dur)
            elif typ == "adc":
                self.adc_count += 1
                if hasattr(obj, "duration"):
                    duration = max(duration, obj.delay + obj.duration)
                else:
                    duration = max(duration, obj.delay + obj.num_samples * obj.dwell)
            elif typ == "delay":
                duration = max(duration, obj.delay)

        # Update total duration
        self._total_duration += duration

        # Get current TR ID
        trid = self.block_trid[self._current_block]
        idx = self.block_within_tr[self._current_block]

        # Assign amplitudes
        self.initial_tr_status[trid][idx, 0] = min(
            self.initial_tr_status[trid][idx, 0], duration
        )
        self.initial_tr_status[trid][idx, 1] = max(
            self.initial_tr_status[trid][idx, 1], rf_amp
        )
        self.initial_tr_status[trid][idx, 2] = max(
            self.initial_tr_status[trid][idx, 2], gx_amp
        )
        self.initial_tr_status[trid][idx, 3] = max(
            self.initial_tr_status[trid][idx, 3], gy_amp
        )
        self.initial_tr_status[trid][idx, 4] = max(
            self.initial_tr_status[trid][idx, 4], gz_amp
        )

        # Assign signs
        if self.gradient_signs[trid][idx, 0] == 0:
            self.gradient_signs[trid][idx, 0] = gx_sign
        if self.gradient_signs[trid][idx, 1] == 0:
            self.gradient_signs[trid][idx, 1] = gy_sign
        if self.gradient_signs[trid][idx, 2] == 0:
            self.gradient_signs[trid][idx, 2] = gz_sign

        # Update position
        self._current_block += 1

    def _add_block_rt(self, *args):
        """Add a block in real-time mode, keeping max rf and gradient amplitudes and minimum duration."""
        if (
            self._current_block >= self._start_block
            and self._current_block < self._end_block
        ):
            args = _harmonize_gradients(*args)
            self._seq.add_block(*args)
        self._current_block += 1


# %% utils
def _harmonize_gradients(*args):
    dummy_seq = PyPulseqSequence(system=Opts(max_grad=np.inf, max_slew=np.inf))
    dummy_seq.add_block(*args)
    block = dummy_seq.get_block(1)
    block = __harmonize_gradients__(block)
    return _block_to_events(block)
