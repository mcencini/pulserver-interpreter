import numpy as np
from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence


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
        self.trid_definitions = (
            {}
        )  # Dict: TRID -> list of TRID pattern for first occurrence
        self._current_trid = None  # The TRID currently being built
        self._block_segment_map = None
        self._seq = PyPulseqSequence(system=system, use_block_cache=use_block_cache)

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
            if trid not in self.trid_definitions:
                self.trid_definitions[trid] = []
                self._current_trid = trid
            else:
                # If we see a TRID > 0 that is already in the definitions, stop building
                self._current_trid = None

        # Only update definition and add blocks if we are building the first instance
        if self._current_trid is not None:
            if trid is not None:
                self.trid_definitions[self._current_trid].append(trid)
            else:
                self.trid_definitions[self._current_trid].append(0)
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
            raise ValueError(f"Mode (={value})must be 'prep', 'eval', or 'rt'.")
        self._mode = value

    def create_segments(self):
        """
        Parse the internal pypulseq sequence libraries, deduplicate events and blocks, and build segment/block mappings.
        Stores the block segment mapping as a private attribute.
        """

        # 1) Parse adc and rf libraries
        seq = self._seq
        if not hasattr(seq, "adc_library") or not seq.adc_library.data:
            raise RuntimeError(
                "Internal pypulseq sequence is empty or missing adc_library."
            )

        # RF: columns 2,3,4,6 (0-based: 1, 2, 3, 5)
        # (mag_id phase_id time_shape_id delay)
        rf_mat = np.stack(list(seq.rf_library.data.values()))[:, [1, 2, 3, 5]]

        # Grad library (0-based: type, 1, 2, 3, 4 + 5 for type == 'g')
        # ('t' rise flat fall delay 0 || 'g' first last amp_shape_id time_shape_id delay)
        grad_types = list(seq.grad_library.type.values())
        grad_data = np.stack(list(seq.grad_library.data.values()))
        grad_mat = []
        for i, t in enumerate(grad_types):
            if t == "g":
                row = [1, *grad_data[i, [1, 2, 3, 4, 5]].tolist()]
            elif t == "t":
                row = [2, *grad_data[i, [1, 2, 3, 4]].tolist(), 0]
            else:
                raise ValueError(f"Unknown grad type: {t}")
            grad_mat.append(row)
        grad_mat = np.asarray(grad_mat)

        # ADC: columns 1,2,3,8 (0-based: 0, 1, 2, 7)
        # (num dwell delay phase_id)
        adc_mat = np.stack(list(seq.adc_library.data.values()))[:, [0, 1, 2, 7]]

        # 2) Find unique rows and mapping for each event type
        _, adc_map = _unique_rows_with_inverse(adc_mat)
        _, rf_map = _unique_rows_with_inverse(rf_mat)
        _, grad_map = _unique_rows_with_inverse(grad_mat)

        # 3) Parse block library and remap event IDs
        block_mat = np.stack(list(seq.block_library.data.values()))
        block_mat = block_mat[:, :6]  # Remove extID column

        # block_mat columns: [duration, rfID, gxID, gyID, gzID, adcID]
        # Remap event IDs to unique event indices
        block_mat[:, 1] = rf_map[block_mat[:, 1]]
        block_mat[:, 2] = grad_map[block_mat[:, 2]]
        block_mat[:, 3] = grad_map[block_mat[:, 3]]
        block_mat[:, 4] = grad_map[block_mat[:, 4]]
        block_mat[:, 5] = adc_map[block_mat[:, 5]]

        # Find unique blocks and mapping
        _, block_map = _unique_rows_with_inverse(block_mat)
        self._block_segment_map = block_map


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
    unique, _, inv = _unique(mat, return_index=True, return_inverse=True)
    return unique, inv
