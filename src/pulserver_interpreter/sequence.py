
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
        self._mode = 'prep'  # 'prep' or 'eval'
        self.trid_definitions = {}  # Dict: TRID -> list of TRID pattern for first occurrence
        self._current_trid = None  # The TRID currently being built
        self._pypulseq_seq = PyPulseqSequence(
            system=system,
            use_block_cache=use_block_cache
        )


    def add_block(self, *args) -> None:
        """
        Add a block to the sequence, dispatching to the appropriate method based on mode.

        Parameters
        ----------
        *args : SimpleNamespace
            Events to add as a block (including possible TRID label as a SimpleNamespace).
        """
        if self._mode == 'prep':
            self._add_block_prep(*args)
        elif self._mode == 'eval':
            self._add_block_eval(*args)
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

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
            if hasattr(obj, 'label') and getattr(obj, 'label', None) == 'TRID':
                trid = getattr(obj, 'value', None)
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
            self._pypulseq_seq.add_block(*args)

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
        return self._pypulseq_seq.system

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ['prep', 'eval', 'rt']:
            raise ValueError(f"Mode (={value})must be 'prep', 'eval', or 'rt'.")
        self._mode = value
