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
        self._current_trid = None  # The TRID currently being built
        self._seq = PyPulseqSequence(system=system, use_block_cache=use_block_cache)
        self._trid_events = {}
        self._segment_library = {}
        self._tr_cursor = {}
        self._segment_cursor = {}
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
            raise ValueError(f"Mode (={value})must be 'prep', 'eval', or 'rt'.")
        self._mode = value

    def create_segments(self):
        """
        Parse the internal pypulseq sequence libraries, deduplicate events and blocks, and build segment/block mappings.
        Stores the block segment mapping as a private attribute.
        Also splits each TR into segments (subarrays between TRID=-1 markers), finds unique segments, and maps segment IDs and block IDs for each TR.
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
        grad_mat = []
        for n, t in seq.grad_library.type.items():
            if t == "g":
                row = [1, *seq.grad_library.data[n][1:6]]
            elif t == "t":
                row = [2, *seq.grad_library.data[n][1:5], 0]
            else:
                raise ValueError(f"Unknown grad type: {t}")
            grad_mat.append(row)
        grad_mat = np.stack(grad_mat)

        # ADC: columns 1,2,3,8 (0-based: 0, 1, 2, 7)
        # (num dwell delay phase_id)
        adc_mat = np.stack(list(seq.adc_library.data.values()))[:, [0, 1, 2, 7]]

        # 2) Find unique rows and mapping for each event type
        _, rf_map = _unique_rows_with_inverse(rf_mat)
        _, grad_map = _unique_rows_with_inverse(grad_mat)
        _, adc_map = _unique_rows_with_inverse(adc_mat)

        # Convert map from 0-indexing to 1-indexing and add 0 for empty events
        rf_map = np.asarray([0, *(rf_map + 1)], dtype=float)
        grad_map = np.asarray([0, *(grad_map + 1)], dtype=float)
        adc_map = np.asarray([0, *(adc_map + 1)], dtype=float)

        # 3) Parse block library and remap event IDs
        block_mat = np.stack(list(seq.block_events.values()))
        block_mat = block_mat[:, :6]  # Remove extID column

        # block_mat columns: [_, rfID, gxID, gyID, gzID, adcID]
        # Remap event IDs to unique event indices
        block_mat[:, 1] = rf_map[block_mat[:, 1]]
        block_mat[:, 2] = grad_map[block_mat[:, 2]]
        block_mat[:, 3] = grad_map[block_mat[:, 3]]
        block_mat[:, 4] = grad_map[block_mat[:, 4]]
        block_mat[:, 5] = adc_map[block_mat[:, 5]]

        # Find pure delay blocks
        is_pure_delay = block_mat.sum(axis=1) == 0

        # Get durations and set duration of pure delay blocks to 0
        durations = np.asarray(list(seq.block_durations.values()), dtype=float)
        durations[is_pure_delay] = 0.0

        # Convert to float and add duration
        block_mat = block_mat.astype(float)
        block_mat[:, 0] = durations

        # Find unique blocks and mapping
        _, block_lut = _unique_rows_with_inverse(block_mat)
        block_lut += 1  # switch to 1-based indexing

        # --- Segment splitting and unique segment detection ---
        # 1. Build a flat list of all block_lut indices for all TRs, and split into TRs
        trid_events = self._trid_events
        tr_block_lut_slices = []  # List of np.array, one per TR
        block_lut_idx = 0
        for _, trdef in trid_events.items():
            tr_len = len(trdef)
            tr_block_lut = block_lut[block_lut_idx : block_lut_idx + tr_len]
            tr_block_lut_slices.append(tr_block_lut)
            block_lut_idx += tr_len

        # 2. For each TR, split into segments at TRID=-1 boundaries (segment starts at -1, not after)
        tr_segments = []  # List of list of np.array (segments per TR)
        tr_segment_starts = []  # List of list of start indices (per TR)
        for _, trdef in trid_events.items():
            tr_blocks = tr_block_lut_slices.pop(0)
            trdef = np.array(trdef)
            # Find indices where TRID == -1
            split_points = np.where(trdef == -1)[0]
            starts = [0, *split_points.tolist()]
            ends = [*split_points.tolist(), len(trdef)]
            segments = [
                tr_blocks[s:e] for s, e in zip(starts, ends, strict=False) if s < e
            ]
            tr_segments.append(segments)
            tr_segment_starts.append(starts)

        # 3. Collect all segments, assign unique segment IDs (1-based), and build segment dict
        segment_dict = {}
        unique_segments = {}
        segment_counter = 1  # Start at 1 for 1-based segment IDs
        for seglist in tr_segments:
            for seg in seglist:
                seg_tuple = tuple(seg.tolist())
                if seg_tuple not in segment_dict:
                    segment_dict[seg_tuple] = segment_counter
                    unique_segments[segment_counter] = seg_tuple
                    segment_counter += 1

        # 4. For each TR, build segmentID and within-segment index arrays, store in dicts keyed by TRID
        tr_segment_ids = {}  # Dict: TRID -> np.array (segment ID for every position)
        tr_block_ids = (
            {}
        )  # Dict: TRID -> np.array (within-segment index for every position)
        for trid, segs in zip(trid_events.keys(), tr_segments, strict=False):
            tr_len = sum(len(seg) for seg in segs)
            seg_id_arr = np.zeros(tr_len, dtype=int)
            within_seg_idx_arr = np.zeros(tr_len, dtype=int)
            pos = 0
            for seg in segs:
                seg_tuple = tuple(seg.tolist())
                seg_id = segment_dict[seg_tuple]
                seg_len = len(seg)
                seg_id_arr[pos : pos + seg_len] = seg_id
                within_seg_idx_arr[pos : pos + seg_len] = np.arange(seg_len)
                pos += seg_len
            tr_segment_ids[trid] = seg_id_arr
            tr_block_ids[trid] = within_seg_idx_arr

        # Store results as attributes
        self._segment_library = (
            unique_segments  # Dict: segment_id (1-based) -> tuple of block IDs
        )
        self._tr_cursor = tr_segment_ids  # Dict: TRID -> np.array, segment IDs per TR (for every position)
        self._segment_cursor = tr_block_ids  # Dict: TRID -> np.array, within-segment index per TR (for every position)

        # Save unique blocks
        block_IDs = np.concatenate(list(unique_segments.values()))
        block_IDs = np.unique(block_IDs)
        for idx in block_IDs:
            self._block_library.add_block(seq.get_block(idx))


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
