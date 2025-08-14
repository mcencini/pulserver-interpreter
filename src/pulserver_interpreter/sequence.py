"""Drop-in replacement for pypulseq.Sequence"""

__all__ = ["Sequence"]

from types import SimpleNamespace
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
        
        self.adc_count = 0
        self._eval_stats = {}
        self._eval_tr_idx = 0
        self._eval_block_idx = 0
        self._eval_trid_list = []
        self._eval_tr_block_counts = {}
        
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
        In eval mode, perform a dry run to determine, for each block within TR, the maximum unsigned values of rf amplitudes and gradient amplitudes on the three axes among all TR instances.
        Uses arrays created during prep step. Raises error if prep was not run.
        Also counts the number of ADC events encountered during the dry run.
        Optimized: directly inspects event objects, no sequence/block creation.
        """
        if not self._trid_events or not self._tr_cursor:
            raise RuntimeError("Eval mode requires a prior prep step with valid TRID events and cursors.")

        if not self._eval_trid_list:
            self._eval_trid_list = list(self._trid_events.keys())
            self._eval_tr_block_counts = {trid: len(self._trid_events[trid]) for trid in self._trid_events}

        curr_trid = self._eval_trid_list[self._eval_tr_idx]
        trsize = self._eval_tr_block_counts[curr_trid]
        if curr_trid not in self._eval_stats:
            arr = np.zeros((trsize, 8), dtype=float)
            arr[:, 2] = 1  # default gx sign +
            arr[:, 4] = 1  # default gy sign +
            arr[:, 6] = 1  # default gz sign +
            arr[:, 0] = np.inf  # duration: start with inf, will take min
            self._eval_stats[curr_trid] = arr
        arr = self._eval_stats[curr_trid]
        idx = self._eval_block_idx
        
        rf_amp = 0.0
        gx = 0.0
        gy = 0.0
        gz = 0.0
        duration = 0.0
        
        # Each obj is an event: check type and extract info
        for obj in args:
            
            # Float: interpreted as block duration
            if isinstance(obj, float):
                duration = max(duration, obj)
                continue
            
            # Event type
            typ = getattr(obj, 'type', None)
            
            # RF event
            if typ == 'rf':
                rf_amp = np.max(np.abs(obj.signal))
                duration = max(duration, obj.delay + obj.shape_dur)
                
            # Gradient events
            elif typ == 'trap':
                if obj.channel == 'x':
                    gx = obj.amplitude
                elif obj.channel == 'y':
                    gy = obj.amplitude
                elif obj.channel == 'z':
                    gz = obj.amplitude
                duration = max(duration, obj.delay + obj.rise_time + obj.flat_time + obj.fall_time + obj.fall_time)
                
            elif typ == 'grad':
                maxval = np.argmax(np.abs(obj.waveform))
                if obj.channel == 'x':
                    gx = obj.waveform[maxval]
                elif obj.channel == 'y':
                    gy = obj.waveform[maxval]
                elif obj.channel == 'z':
                    gz = obj.waveform[maxval]
                duration = max(duration, obj.delay + obj.shape_dur)
                
            # ADC event
            elif typ == 'adc':
                self.adc_count += 1
                if hasattr(obj, 'duration'):
                    duration = max(duration, obj.delay + obj.duration)
                else:
                    duration = max(duration, obj.delay + obj.num_samples * obj.dwell)
                    
            # Delay
            elif typ == 'delay':
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
        
        
# %% utils
def _get_block(*args):
    _seq = PyPulseqSequence(system=Opts(max_grad=np.inf, max_slew=np.inf))
    _seq.add_block(*args)
    return _seq.get_block(1)