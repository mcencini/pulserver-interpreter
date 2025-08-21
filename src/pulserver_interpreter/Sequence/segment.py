"""Segment identification and global array population."""

__all__ = ["get_seq_structure"]

from types import SimpleNamespace

import numpy as np

from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence


def get_seq_structure(
    seq: PyPulseqSequence,
    first_tr_instances_trid_labels: dict | None = None,
    trid_labels: list | None = None,
) -> tuple[
    dict, dict, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Build segments, TR definitions, and global arrays efficiently.

    Returns
    -------
    trs : dict
        Mapping from TRID to list of segment IDs composing it.
    segments : dict
        Mapping from segment_id to array of unique block IDs.
    unique_blocks : dict
        Dictionary containing the unique blocks used in `segments`.
    block_tr : np.ndarray
        Array mapping each block to its TR ID.
    block_within_tr : np.ndarray
        Array mapping each block to its position within the TR.
    block_segment_id : np.ndarray
        Array mapping each block to its segment ID.
    block_within_segment : np.ndarray
        Array mapping each block to its position within the segment.
    block_id : np.ndarray
        Array of global block IDs for each block in sequence order.
    """
    trid_labels = np.asarray(trid_labels) if trid_labels is not None else None

    # Step 0: autotr not allowed for not
    if trid_labels is None:
        raise RuntimeError("Please provide manual TR definitions")

    # Step 1: Deduplicate blocks
    unique_block_ids, block_lut = deduplicate_blocks(seq)

    # Store unique blocks in a Sequence object
    dummy_sys = Opts(max_grad=np.inf, max_slew=np.inf)
    unique_blocks = PyPulseqSequence(system=dummy_sys)

    for idx in unique_block_ids:
        block = seq.get_block(idx)
        unique_blocks.add_block(block)

    # Remap block lut
    block_lut = np.searchsorted(unique_block_ids, block_lut) + 1

    if trid_labels is None or np.all(trid_labels == 0):
        block_id = block_lut.copy()
    else:
        block_id = None

    # Step 2: TR splitting
    if first_tr_instances_trid_labels is None:
        block_lut, first_tr_instances_trid_labels = autotr(block_lut)

    # Step 3: Segment splitting
    has_trid_minus1 = np.any(trid_labels == -1) if trid_labels is not None else False
    if not has_trid_minus1:
        segments, trs = autoseg(block_lut, first_tr_instances_trid_labels)
    else:
        segments, trs = split_segments(block_lut, first_tr_instances_trid_labels)

    # Step 4: Deduplicate segments
    segments, trs = deduplicate_segments(segments, trs)

    # Step 5: Build global arrays
    # tr starts and tr ids
    n_total_segments = np.sum(trid_labels != 0).item()
    trid_labels = trid_labels.copy()
    trid_labels[trid_labels <= 0] = 0  #  erase segment breaks

    # Get tr starts
    tr_starts = trid_labels > 0

    # Forward fill block  TRID
    block_trid = np.maximum.accumulate(trid_labels)

    # Cumulative counter over the sequence
    counter = np.arange(len(block_trid))

    # Reset per TR by subtracting the start index of the current TR
    # Use maximum.accumulate on start indices
    start_idx = np.where(tr_starts, np.arange(len(block_trid)), 0)
    start_idx = np.maximum.accumulate(start_idx)
    block_within_tr = counter - start_idx

    # segment id
    trid_labels = trid_labels[trid_labels > 0]

    # Get length of each segment
    segment_lengths_dict = {seg_id: len(blocks) for seg_id, blocks in segments.items()}

    # Get whole sequence of segment IDs and matching sizes
    segment_ids = np.concatenate([trs[t].segments for t in trid_labels])
    segment_lengths = np.array([segment_lengths_dict[s] for s in segment_ids])
    block_segment_id = np.repeat(segment_ids, segment_lengths)
    block_within_segment = np.concatenate(
        [np.arange(segment_lengths_dict[s]) for s in segment_ids]
    )

    # block id
    block_id = np.concatenate([trs[t].blocks for t in trid_labels])

    return (
        trs,
        segments,
        unique_blocks,
        block_trid,
        block_within_tr,
        block_segment_id,
        block_within_segment,
        block_id,
        n_total_segments,
    )


# %% ================= Helper functions ===================
def deduplicate_blocks(seq: PyPulseqSequence) -> np.ndarray:
    """
    Parse sequence and build unique block ID array.

    Parameters
    ----------
    seq : PyPulseqSequence
        Input sequence object.

    Returns
    -------
    block_lut : np.ndarray
        1D array of unique block IDs (1-based indexing).

    """
    # RF: columns 2,3,4,6 (0-based: 1,2,3,5)
    rf_mat = np.stack(list(seq.rf_library.data.values()))[:, [1, 2, 3, 5]]

    # Grad library
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

    # ADC
    adc_mat = np.stack(list(seq.adc_library.data.values()))[:, [0, 1, 2, 7]]

    # Unique rows
    _, rf_map = _unique_rows_with_inverse(rf_mat)
    _, grad_map = _unique_rows_with_inverse(grad_mat)
    _, adc_map = _unique_rows_with_inverse(adc_mat)

    # Remap to 1-based, add 0 for empty
    rf_map = np.asarray([0, *(rf_map + 1)], dtype=float)
    grad_map = np.asarray([0, *(grad_map + 1)], dtype=float)
    adc_map = np.asarray([0, *(adc_map + 1)], dtype=float)

    # Parse block library
    block_mat = np.stack(list(seq.block_events.values()))[:, :6]
    block_mat[:, 2] = grad_map[block_mat[:, 2]]
    block_mat[:, 3] = grad_map[block_mat[:, 3]]
    block_mat[:, 4] = grad_map[block_mat[:, 4]]
    block_mat[:, 5] = adc_map[block_mat[:, 5]]

    is_pure_delay = block_mat.sum(axis=1) == 0
    durations = np.asarray(list(seq.block_durations.values()), dtype=float)
    durations[is_pure_delay] = 0.0
    block_mat = block_mat.astype(float)
    block_mat[:, 0] = durations

    unique_block_id, block_lut = _unique_rows_with_inverse(block_mat)

    return unique_block_id + 1, block_lut + 1  # 1-based indexing


def autotr(block_lut: np.ndarray) -> dict:
    """
    Prototype for automatic TR detection.

    Parameters
    ----------
    block_lut : np.ndarray
        1D array of unique block IDs.

    Returns
    -------
    trid_labels : dict
        TRID labels for each block within the first instance
        of each TR.

    """
    raise NotImplementedError("autotr is not yet implemented.")


def autoseg(block_lut: np.ndarray, trid_labels: dict) -> tuple[dict, dict]:
    """
    Prototype for automatic segment detection within a TR.

    Parameters
    ----------
    block_lut : np.ndarray
        1D array of block IDs.
    trid_labels : dict
        TRID labels for each block within the first instance
        of each TR. Example: ``{1: [1, 0, -1, 0, 0]}``.

    Returns
    -------
    segments : dict
        Mapping from segment_id (int) to array of block IDs.
    trs : dict
        Mapping from TRID to list of segment IDs composing that TR in order.

    """
    raise NotImplementedError("autoseg is not yet implemented.")


def split_segments(block_lut: np.ndarray, trid_labels: dict) -> tuple[dict, dict]:
    """
    Split each TR into segments at TRID=-1 boundaries and return both
    segment definitions and TR compositions.

    Parameters
    ----------
    block_lut : np.ndarray
        1D array of global block IDs.
    trid_labels : dict
        TRID labels for each block within the first instance
        of each TR. Example: ``{1: [1, 0, -1, 0, 0]}``.

    Returns
    -------
    segments : dict
        Mapping from segment_id (int) to array of block IDs.
    trs : dict
        Mapping from TRID to a SimpleNamespace with:

        - ``segments`` : list of segment IDs composing the TR
        - ``blocks``   : array of block IDs composing the TR in order

    """
    segments = {}
    trs = {}
    segment_counter = 1

    block_ptr = 0
    for trid, trdef in trid_labels.items():
        trdef = np.array(trdef)
        tr_len = len(trdef)
        tr_blocks = block_lut[block_ptr : block_ptr + tr_len]
        block_ptr += tr_len

        # find split points
        split_points = np.where(trdef == -1)[0]
        starts = [0, *split_points.tolist()]
        ends = [*split_points.tolist(), len(trdef)]

        segment_ids = []
        for s, e in zip(starts, ends, strict=False):
            if s >= e:
                continue
            seg_blocks = tr_blocks[s:e]
            segments[segment_counter] = seg_blocks
            segment_ids.append(segment_counter)
            segment_counter += 1

        trs[trid] = SimpleNamespace(segments=segment_ids, blocks=tr_blocks)

    return segments, trs


def deduplicate_segments(segments: dict, trs: dict) -> tuple[dict, dict]:
    """
    Deduplicate segments and update TR definitions accordingly.

    Parameters
    ----------
    segments : dict
        Mapping from segment_id (int) to array of block IDs.
    trs : dict
        Mapping from TRID to a SimpleNamespace with:

        - ``segments`` : list of segment IDs composing the TR
        - ``blocks``   : array of block IDs composing the TR in order

    Returns
    -------
    unique_segments : dict
        Deduplicated mapping from segment_id (int) to array of block IDs.
    updated_trs : dict
        Same structure as input `trs`, but with ``segments`` remapped
        to deduplicated segment IDs.

    """
    segment_dict = {}  # maps tuple(blocks) → new segment_id
    unique_segments = {}  # new deduplicated segment dictionary
    seg_remap = {}  # maps old segment_id → new segment_id
    segment_counter = 1

    # Pass 1: deduplicate segments
    for seg_id, seg_blocks in segments.items():
        seg_tuple = tuple(seg_blocks.tolist())
        if seg_tuple not in segment_dict:
            segment_dict[seg_tuple] = segment_counter
            unique_segments[segment_counter] = seg_blocks
            seg_remap[seg_id] = segment_counter
            segment_counter += 1
        else:
            seg_remap[seg_id] = segment_dict[seg_tuple]

    # Pass 2: remap TR segment definitions
    updated_trs = {}
    for trid, tr in trs.items():
        new_segment_ids = [seg_remap[sid] for sid in tr.segments]
        updated_trs[trid] = SimpleNamespace(
            segments=new_segment_ids, blocks=tr.blocks.copy()
        )

    return unique_segments, updated_trs


# %% ==== Utilities ====
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
    return np.sort(idx), idx[inv]
