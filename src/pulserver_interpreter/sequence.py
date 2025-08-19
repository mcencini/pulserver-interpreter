"""Drop-in replacement for pypulseq.Sequence with TR/segment tracking"""

__all__ = ["Sequence"]


from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence

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
        Add a block in preparation mode, tracking TRID, within-TR index, and first TR instance labels.
        """
        if self.prepped:
            raise ValueError(
                "Sequence is already prepared. Please call clear() running another preparation pass"
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
        """
        Placeholder for eval mode.
        """
        raise NotImplementedError("Eval mode not implemented yet.")

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

    def get_seq_structure(self):
        """
        Use the external segment.build_segments wrapper to perform block deduplication, segment splitting, and segment deduplication.
        Stores the resulting segment library, mapping arrays, and block library as attributes.
        """
        if self.prepped:
            raise ValueError(
                "Sequence is already prepared. Please call clear() before parsing structure again"
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
        ) = _get_seq_structure(
            self._seq, self.first_tr_instances_trid_labels, self.block_trid
        )
        self.prepped = True
