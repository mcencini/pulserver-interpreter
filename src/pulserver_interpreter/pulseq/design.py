"""Base class for Pylseq design routines."""

__all__ = ["CompositeDesign", "PulseqDesign", "concatenate"]

from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np

from pypulseq import Opts

from .Sequence.sequence import Sequence


class PulseqDesign(ABC):
    """
    Base class for Pulseq design routines with support for
    dry-mode outer loop scaling and default argument storage.

    Subclasses should implement `core(self, *args, **kwargs)`.
    The design should place its outer loop inside `self.range(N)`.
    """

    def __init__(
        self, system: Opts | None = None, use_block_cache: bool = True, **defaults
    ):
        """
        Initialize a new Sequence instance and store default arguments.

        Parameters
        ----------
        system : Opts, optional
            System limits or configuration.
        use_block_cache : bool, optional
            Whether to use block cache (default: True).
        defaults : dict
            Default arguments for the core method.

        """
        self.seq = Sequence(system, use_block_cache)
        self.seqID = 1
        self._mode = "dry"  # 'dry', 'prep', 'eval','rt'
        self._singleton = True
        self._range_used = False
        self._outer_iterations = 1
        self.__standalone__ = True
        self._defaults = defaults
        self._start_block = 0
        self._end_block = np.inf

    @abstractmethod
    def core(self, *args, **kwargs):
        """Subclasses must implement this method."""
        ...

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["dry", "prep", "eval", "rt"]:
            raise ValueError(f"Mode (={value}) must be 'dry', 'prep', 'eval', or 'rt'.")
        self.seq._mode = value
        self._mode = value

    @property
    def start_block(self):
        return self._start_block

    @start_block.setter
    def start_block(self, value: int):
        self.seq.start_block = value
        self._start_block = value

    @property
    def end_block(self):
        return self._end_block

    @end_block.setter
    def end_block(self, value: int):
        self.seq.end_block = value
        self._end_block = value

    def __call__(self, *args, **kwargs):
        """Run the design routine."""
        # Merge defaults with call-specific kwargs
        merged_kwargs = {**self._defaults, **kwargs}

        self._range_used = False
        self._outer_iterations = 1

        if self.mode == "dry":
            self._singleton = True
            if self.__standalone__:
                self.seq.clear()
            self.core(*args, **merged_kwargs)

            # get rfIDs in block events
            rfIDs = list(self.seq.seq.block_events.values())
            rfIDs = np.stack(rfIDs)[:, 1]
            rfIDs = rfIDs[np.where(rfIDs != 0)]

            # Amp + raw pulse shape IDs
            rf = [self.seq.seq.rf_library.data[rfID][:4] for rfID in rfIDs]

            return SimpleNamespace(
                duration=self.seq._total_duration * self._outer_iterations,
                n_total_segments=self.seq.n_total_segments * self._outer_iterations,
                rf=rf,
                tr=self.seq._total_duration,
            )

        if self.mode == "prep":
            self._singleton = False
            if self.__standalone__:
                self.seq.clear()
            self.seq.mode = "prep"
            self.core(*args, **merged_kwargs)
            if self.__standalone__:
                self.seq.get_seq_structure()

        if self.mode == "eval":
            self.seq.mode = "eval"
            self.core(*args, **merged_kwargs)
            if self.__standalone__:
                self.seq.get_initial_status()

        if self.mode == "rt":
            self._singleton = False
            if self.__standalone__:
                self.seq.clear_buffer()
            self.seq.mode = "rt"
            self.core(*args, **kwargs)

        return self.seq

    def range(self, n: int):
        """
        Special replacement for Python range().

        In normal mode -> acts like built-in range(n).
        In dry mode    -> runs a single iteration, then records n
                          so we can scale durations afterwards.
        """
        if self._singleton:
            if self._range_used:
                raise RuntimeError(
                    "self.range can only be called once per core() execution"
                )
            self._range_used = True
            self._outer_iterations = n
            return range(1)
        return range(n)


class CompositeDesign:
    """
    Container for multiple PulseqDesigns.
    Creates a shared Sequence and injects it into child designs.
    """

    def __init__(self, *designs: PulseqDesign):
        self._mode = "dry"
        self.designs = list(designs)

        # Shared sequence
        self.seq = Sequence(designs[0].seq.system, designs[0].seq.use_block_cache)

        # Replace child sequences with shared one
        for n, d in enumerate(self.designs, start=1):
            d.seq = self.seq
            d.__standalone__ = False
            d.seqID = n

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["dry", "prep", "eval", "rt"]:
            raise ValueError(f"Mode (={value}) must be 'dry', 'prep', 'eval', or 'rt'.")
        for design in self.designs:
            design.mode = value
        self._mode = value

    @property
    def start_block(self):
        return self._start_block

    @start_block.setter
    def start_block(self, value: int):
        for design in self.designs:
            design.start_block = value
        self._start_block = value

    @property
    def end_block(self):
        return self._end_block

    @end_block.setter
    def end_block(self, value: int):
        for design in self.designs:
            design.end_block = value
        self._end_block = value

    def __call__(self, *args, **kwargs):
        """
        Executes all child designs in sequence.
        Only keyworded arguments are allowed,
        each must be a dict for a specific child design.

        Parameters
        ----------
        kwargs : dict
            Optional argument overrides for each design, e.g.
            combo(gre=dict(Ny=128), mprage=dict(TR=500))
        """
        if args:
            raise TypeError(
                "CompositeDesign only accepts keyword arguments, "
                "one per sub-design, e.g. combo(gre=dict(Ny=128), mprage=dict(TR=500))"
            )

        results = []

        if self.mode in ["dry", "prep"]:
            self.seq.clear()
        elif self.mode == "rt":
            self.seq.clear_buffer()

        for design in self.designs:
            design.mode = self.mode

        if self.mode == "dry":
            for design in self.designs:
                design_kwargs = kwargs.get(design.__class__.__name__.lower(), {})
                results.append(design(**design_kwargs))

            return SimpleNamespace(
                duration=np.sum([res.duration for res in results]).item(),
                n_total_segments=np.sum(
                    [res.n_total_segments for res in results]
                ).item(),
                rf=[res.rf for res in results],
                tr=[res.tr for res in results],
            )

        # Prep / real-time execution
        for design in self.designs:
            design_kwargs = kwargs.get(design.__class__.__name__.lower(), {})
            design(**design_kwargs)

        if self.mode == "prep":
            self.seq.get_seq_structure()

        if self.mode == "eval":
            self.seq.get_initial_status()

        return self.seq


def concatenate(*designs: list[PulseqDesign]) -> CompositeDesign:
    """
    Concatenate multiple sequence designers.

    Parameters
    ----------
    *designs : PulseqDesign
        Arbitrary length sequence design objects.

    Returns
    -------
    CompositeDesign
        Composite design function.

    """
    return CompositeDesign(*designs)
