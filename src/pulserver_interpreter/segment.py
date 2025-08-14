"""Segment identifications subroutines."""

__all__ = ["build_segments"]

import numpy as np
from pypulseq import Sequence as PyPulseqSequence


def build_segments(
    seq: PyPulseqSequence, trid_events: dict | None = None
) -> tuple[dict, dict, dict, dict, PyPulseqSequence]:
    """
    Perform block deduplication, segment splitting, and segment deduplication.

    Parameters
    ----------
    seq : PyPulseqSequence
        The input sequence object.
    trid_events : dict, optional
        Dictionary mapping TRID to list of TRID events. If None or has no TRID > 0, auto-TR detection is used.

    Returns
    -------
    segment_library : dict
        Mapping from segment_id to tuple of block IDs.
    tr_cursor : dict
        Mapping from TRID to np.ndarray of segment IDs per TR (for every position).
    segment_cursor : dict
        Mapping from TRID to np.ndarray of within-segment index per TR (for every position).
    trid_definitions : dict
        Mapping from TRID to tuple of segment IDs (in order).
    block_library : PyPulseqSequence
        Sequence containing all unique blocks.
    """
    block_lut = deduplicate_blocks(seq)

    # Step 1: TR detection
    if trid_events is None or not any(
        any(e > 0 for e in v) for v in trid_events.values()
    ):
        # No TRID > 0 present, use autotr
        trid_events = autotr(block_lut)

    # Step 2: Segment splitting
    has_trid_minus1 = any(-1 in v for v in trid_events.values())
    if not has_trid_minus1:
        # No TRID == -1 present, use autoseg
        tr_segments = autoseg(trid_events, block_lut)
    else:
        tr_segments = split_segments(trid_events, block_lut)

    # Step 3: Segment deduplication and cursor/library construction
    segment_library, tr_cursor, segment_cursor, trid_definitions = deduplicate_segments(
        tr_segments
    )

    # Save unique blocks to a new PyPulseqSequence
    block_library = PyPulseqSequence(system=seq.system)
    block_IDs = np.concatenate(list(segment_library.values()))
    block_IDs = np.unique(block_IDs)
    for idx in block_IDs:
        block_library.add_block(seq.get_block(idx))

    return segment_library, tr_cursor, segment_cursor, trid_definitions, block_library


def deduplicate_blocks(seq: PyPulseqSequence) -> np.ndarray:
    """
    Parse the internal pypulseq sequence libraries and build a unique block ID array.

    Parameters
    ----------
    seq : PyPulseqSequence
        The input sequence object.

    Returns
    -------
    block_lut : np.ndarray
        1D array of unique block IDs (1-based indexing).
    """
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

    # Find unique rows and mapping for each event type
    _, rf_map = _unique_rows_with_inverse(rf_mat)
    _, grad_map = _unique_rows_with_inverse(grad_mat)
    _, adc_map = _unique_rows_with_inverse(adc_mat)

    # Convert map from 0-indexing to 1-indexing and add 0 for empty events
    rf_map = np.asarray([0, *(rf_map + 1)], dtype=float)
    grad_map = np.asarray([0, *(grad_map + 1)], dtype=float)
    adc_map = np.asarray([0, *(adc_map + 1)], dtype=float)

    # Parse block library and remap event IDs
    block_mat = np.stack(list(seq.block_events.values()))
    block_mat = block_mat[:, :6]  # Remove extID column

    # block_mat columns: [_, rfID, gxID, gyID, gzID, adcID]
    # Remap event IDs to unique event indices
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

    return block_lut


def autotr(block_lut: np.ndarray) -> dict:
    """
    Prototype for automatic TR detection.

    Parameters
    ----------
    block_lut : np.ndarray
        1D array of unique block IDs.

    Returns
    -------
    trid_events : dict
        Dictionary mapping TRID to list of TRID events (synthetic TRID array for each TR).
    """
    # TODO: Implement periodicity/pattern detection
    raise NotImplementedError("autotr is not yet implemented.")


def autoseg(trid_events: dict, block_lut: np.ndarray) -> list:
    """
    Prototype for automatic segment detection within a TR.

    Parameters
    ----------
    trid_events : dict
        Dictionary mapping TRID to list of TRID events.
    block_lut : np.ndarray
        1D array of unique block IDs.

    Returns
    -------
    tr_segments : list
        List of segment arrays per TR.
    """
    # TODO: Implement segment boundary detection heuristics
    raise NotImplementedError("autoseg is not yet implemented.")


def split_segments(trid_events: dict, block_lut: np.ndarray) -> list:
    """
    Split each TR into segments at TRID=-1 boundaries.

    Parameters
    ----------
    trid_events : dict
        Dictionary mapping TRID to list of TRID events.
    block_lut : np.ndarray
        1D array of unique block IDs.

    Returns
    -------
    tr_segments : list
        List of segment arrays per TR.
    """
    tr_block_lut_slices: list = []
    block_lut_idx = 0
    for _, trdef in trid_events.items():
        tr_len = len(trdef)
        tr_block_lut = block_lut[block_lut_idx : block_lut_idx + tr_len]
        tr_block_lut_slices.append(tr_block_lut)
        block_lut_idx += tr_len
    tr_segments = []
    for _, trdef in trid_events.items():
        tr_blocks = tr_block_lut_slices.pop(0)
        trdef = np.array(trdef)
        split_points = np.where(trdef == -1)[0]
        starts = [0, *split_points.tolist()]
        ends = [*split_points.tolist(), len(trdef)]
        segments = [tr_blocks[s:e] for s, e in zip(starts, ends, strict=False) if s < e]
        tr_segments.append(segments)
    return tr_segments


def deduplicate_segments(tr_segments: list) -> tuple[dict, dict, dict, dict]:
    """
    Deduplicate segments, assign segment IDs, and build segment library and cursors.

    Parameters
    ----------
    tr_segments : list
        List of segment arrays per TR.

    Returns
    -------
    segment_library : dict
        Mapping from segment_id to tuple of block IDs.
    tr_cursor : dict
        Mapping from TRID to np.ndarray of segment IDs per TR (for every position).
    segment_cursor : dict
        Mapping from TRID to np.ndarray of within-segment index per TR (for every position).
    trid_definitions : dict
        Mapping from TRID to tuple of segment IDs (in order).
    """
    # Collect all segments, assign unique segment IDs (1-based), and build segment dict
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

    # For each TR, build segmentID and within-segment index arrays, store in dicts keyed by TRID
    tr_segment_ids = {}
    tr_block_ids = {}
    tr_library = {}
    for trid, segs in zip(range(1, len(tr_segments) + 1), tr_segments, strict=False):
        tr_len = sum(len(seg) for seg in segs)
        seg_id_arr = np.zeros(tr_len, dtype=int)
        within_seg_idx_arr = np.zeros(tr_len, dtype=int)
        pos = 0
        seg_ids = []
        for seg in segs:
            seg_tuple = tuple(seg.tolist())
            seg_id = segment_dict[seg_tuple]
            seg_len = len(seg)
            seg_id_arr[pos : pos + seg_len] = seg_id
            within_seg_idx_arr[pos : pos + seg_len] = np.arange(seg_len)
            pos += seg_len
            seg_ids.append(seg_id)
        tr_segment_ids[trid] = seg_id_arr
        tr_block_ids[trid] = within_seg_idx_arr
        tr_library[trid] = tuple(seg_ids)
    return unique_segments, tr_segment_ids, tr_block_ids, tr_library


# %% utils
def _unique(
    arr: np.ndarray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple | np.ndarray:
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


def _unique_rows_with_inverse(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, idx, inv = _unique(mat, return_index=True, return_inverse=True)
    return mat[np.sort(idx)], idx[inv]
